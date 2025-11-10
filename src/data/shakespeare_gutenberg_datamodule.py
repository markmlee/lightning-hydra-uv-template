"""DataModule for the Project Gutenberg Shakespeare corpus from HuggingFace.

This module handles loading and preprocessing of the swiss-ai/apertus-pretrain-gutenberg
dataset, which contains a large corpus of classical literature including Shakespeare.
"""

from typing import Any

import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ShakespeareGutenbergDataModule(LightningDataModule):
    """DataModule for Project Gutenberg Shakespeare corpus.

    Loads the swiss-ai/apertus-pretrain-gutenberg dataset and tokenizes it
    for causal language modeling with SmolLM.

    Args:
        tokenizer_name: Name of the HuggingFace tokenizer to use
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        train_val_test_split: Tuple of (train, val, test) split ratios
        max_samples: Maximum number of samples to use (None = use all). Useful for testing.

    """

    def __init__(
        self,
        tokenizer_name: str = "HuggingFaceTB/SmolLM-135M",
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        train_val_test_split: tuple[float, float, float] = (0.95, 0.025, 0.025),
        max_samples: int | None = None,
    ) -> None:
        """Initialize the datamodule."""
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Ensure padding token is set for causal LM
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Called only on 1 GPU/TPU in distributed."""
        # Download tokenizer (dataset will be loaded via streaming in setup)
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

    def setup(self, stage: str | None = None) -> None:  # noqa: C901
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning on every process when using DDP.
        """
        # Divide batch size by the number of devices
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # Load dataset splits
        if not self.data_train and not self.data_val and not self.data_test:
            # Load dataset - using streaming to handle mixed formats
            print("Loading Gutenberg dataset (this may take a few minutes)...")
            dataset = load_dataset(
                "swiss-ai/apertus-pretrain-gutenberg",
                split="train",
                streaming=True,  # Use streaming to handle large dataset
                trust_remote_code=True,
            )

            # Filter to only keep entries with 'text' field (not pre-tokenized)
            dataset = dataset.filter(lambda x: "text" in x and "input_ids" not in x)

            # Convert to regular dataset (take samples)
            if self.hparams.max_samples is not None:
                dataset = dataset.take(self.hparams.max_samples)
            else:
                # For full dataset, take a large but manageable number
                dataset = dataset.take(50000)  # Limit to 50k for memory efficiency

            # Convert to list and then to regular dataset
            dataset_list = list(dataset)
            from datasets import Dataset as HFDataset

            dataset = HFDataset.from_dict(
                {"text": [item["text"] for item in dataset_list]}
            )
            print(f"Loaded {len(dataset)} samples from Gutenberg corpus")

            # Tokenize documents individually to avoid memory issues
            # For 50k samples with 1.6B+ characters, tokenizing all at once causes OOM
            print("Tokenizing documents in chunks (this may take a few minutes)...")
            all_input_ids_list = []

            # Process in batches to show progress and avoid memory issues
            batch_size = 100
            for i in range(0, len(dataset), batch_size):
                batch_texts = dataset["text"][i : i + batch_size]
                batch_text = " ".join(batch_texts)

                # Tokenize this batch
                tokenized_batch = self.tokenizer(
                    batch_text,
                    truncation=False,
                    return_tensors="pt",
                )
                all_input_ids_list.append(tokenized_batch["input_ids"][0])

                # Progress update
                if (i // batch_size) % 10 == 0:
                    print(f"  Processed {i}/{len(dataset)} documents...")

            # Concatenate all tokenized batches
            print("Concatenating tokenized batches...")
            all_input_ids = torch.cat(all_input_ids_list, dim=0)
            print(f"Total tokens: {len(all_input_ids):,}")
            del all_input_ids_list  # Free memory

            # Split into fixed-length sequences (non-overlapping chunks)
            print(f"Creating sequences of length {self.hparams.max_length}...")
            sequences = []
            for i in range(
                0, len(all_input_ids) - self.hparams.max_length, self.hparams.max_length
            ):
                seq = all_input_ids[i : i + self.hparams.max_length]
                sequences.append(seq)
            print(f"Created {len(sequences)} sequences")

            # Stack into tensor
            input_ids_tensor = torch.stack(sequences)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)

            # Create PyTorch Dataset
            class TokenizedDataset(Dataset):
                def __init__(
                    self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> None:
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask

                def __len__(self) -> int:
                    return len(self.input_ids)

                def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                    return {
                        "input_ids": self.input_ids[idx],
                        "attention_mask": self.attention_mask[idx],
                        "labels": self.input_ids[
                            idx
                        ],  # For causal LM, labels = input_ids
                    }

            full_dataset = TokenizedDataset(input_ids_tensor, attention_mask_tensor)

            # Split into train/val/test
            total_size = len(full_dataset)
            train_size = int(total_size * self.hparams.train_val_test_split[0])
            val_size = int(total_size * self.hparams.train_val_test_split[1])

            self.data_train = torch.utils.data.Subset(
                full_dataset, range(0, train_size)
            )
            self.data_val = torch.utils.data.Subset(
                full_dataset, range(train_size, train_size + val_size)
            )
            self.data_test = torch.utils.data.Subset(
                full_dataset, range(train_size + val_size, total_size)
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after fit/val/test/predict."""
        pass

    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint. Implement to generate and save datamodule state."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state."""
        pass


if __name__ == "__main__":
    # Test the datamodule
    dm = ShakespeareGutenbergDataModule(max_samples=1000)  # Limit for testing
    dm.prepare_data()
    dm.setup()
    print(f"Tokenizer name: {dm.hparams.tokenizer_name}")
    print(f"Tokenizer type: {type(dm.tokenizer)}")

    if dm.data_train is None or dm.data_val is None or dm.data_test is None:
        raise RuntimeError("Data not loaded properly")

    print(f"Train dataset size: {len(dm.data_train)}")
    print(f"Val dataset size: {len(dm.data_val)}")
    print(f"Test dataset size: {len(dm.data_test)}")

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print("\n✓ Gutenberg DataModule working correctly!")
