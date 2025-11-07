from typing import Any

import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TinyShakespeareDataModule(LightningDataModule):
    """`LightningDataModule` for the TinyShakespeare dataset.

    The TinyShakespeare dataset contains approximately 40,000 lines of Shakespeare text,
    commonly used for character-level or word-level language modeling tasks.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        tokenizer_name: str = "HuggingFaceTB/SmolLM-135M",
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_val_test_split: tuple[float, float, float] = (0.9, 0.05, 0.05),
    ) -> None:
        """Initialize a `TinyShakespeareDataModule`.

        :param tokenizer_name: The name of the HuggingFace tokenizer to use.
        :param max_length: Maximum sequence length for tokenization.
        :param batch_size: The batch size. Defaults to `8`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param persistent_workers: Whether to keep workers alive between epochs.
        :param train_val_test_split: The train, validation and test split ratios.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed.

        Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Download the dataset
        load_dataset("tiny_shakespeare", trust_remote_code=True)

        # Download the tokenizer
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Load the dataset
            dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)

            # The tiny_shakespeare dataset structure: each split has 1 row with full text
            # We'll use the train split which contains the full Shakespeare text
            full_text = dataset["train"]["text"][0]  # Get the single text string

            # Tokenize the entire text without padding to get all tokens
            tokenized = self.tokenizer(
                full_text,
                truncation=False,
                return_tensors="pt",
            )

            # Get the full sequence of token IDs
            all_input_ids = tokenized["input_ids"][0]

            # Chunk the tokens into sequences of max_length
            sequences = []
            for i in range(
                0, len(all_input_ids) - self.hparams.max_length, self.hparams.max_length
            ):
                seq = all_input_ids[i : i + self.hparams.max_length]
                sequences.append(seq)

            # Convert sequences to tensors
            input_ids_tensor = torch.stack(sequences)

            # Create attention masks (all ones since we have no padding)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)

            # Create dataset from tokenized data
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
                        "labels": self.input_ids[idx],  # For causal language modeling
                    }

            full_dataset = TokenizedDataset(input_ids_tensor, attention_mask_tensor)

            # Calculate split sizes
            total_size = len(full_dataset)
            train_size = int(total_size * self.hparams.train_val_test_split[0])
            val_size = int(total_size * self.hparams.train_val_test_split[1])

            # Split the dataset
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
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up.

        Cleanup after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    # Simple test
    dm = TinyShakespeareDataModule()
    dm.prepare_data()
    dm.setup()
    print(f"Tokenizer name: {dm.hparams.tokenizer_name}")
    print(f"Tokenizer type: {type(dm.tokenizer)}")

    if dm.data_train is None or dm.data_val is None or dm.data_test is None:
        raise RuntimeError("Data not loaded properly")

    print(f"Train dataset size: {len(dm.data_train)}")
    print(f"Val dataset size: {len(dm.data_val)}")
    print(f"Test dataset size: {len(dm.data_test)}")

    # Test a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
