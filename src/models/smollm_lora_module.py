"""SmolLM LightningModule with LoRA (Low-Rank Adaptation) for efficient fine-tuning.

This module implements parameter-efficient fine-tuning using LoRA, which adds small
trainable adapter matrices to the frozen pre-trained model. This significantly reduces
memory usage and training time while maintaining good performance.
"""

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric
from transformers import AutoConfig, AutoModelForCausalLM


class SmolLMLoRALitModule(LightningModule):
    """LightningModule for SmolLM with LoRA adapters.

    Wraps HuggingFace's AutoModelForCausalLM and applies LoRA for efficient fine-tuning.
    Only the LoRA adapter weights are trained, while the base model remains frozen.

    Args:
        model_name: HuggingFace model name (e.g., "HuggingFaceTB/SmolLM-135M")
        lora_r: LoRA rank (dimensionality of adapter matrices)
        lora_alpha: LoRA scaling parameter (typically 2*r)
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: List of module names to apply LoRA to
        optimizer: Optimizer configuration (partial, instantiated by Lightning)
        scheduler: Learning rate scheduler configuration (partial, optional)
        compile: Whether to compile the model with torch.compile (PyTorch 2.0+)
        dropout: Dropout probability for model (applied via config)

    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        compile: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the LoRA module."""
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load model config and set dropout
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "attention_dropout"):
            config.attention_dropout = dropout
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout
        if hasattr(config, "resid_pdrop"):
            config.resid_pdrop = dropout
        if hasattr(config, "attn_pdrop"):
            config.attn_pdrop = dropout

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
        )

        # Configure LoRA
        # Default target modules for Llama-based models (SmolLM uses Llama architecture)
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters summary
        self.model.print_trainable_parameters()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MeanMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for loss computation (batch_size, seq_len)

        Returns:
            Dictionary containing loss and logits

        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels

        Returns:
            Tuple of (loss, logits)

        """
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs["loss"], outputs["logits"]

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels
            batch_idx: Index of the current batch

        Returns:
            Loss tensor

        """
        loss, _logits = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)

        # Log step-level loss for wandb (shows progress during epoch)
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)

        # Log learning rate at step level (crucial for debugging LR schedule issues!)
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(
                "train/lr_step",
                current_lr,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        # Log epoch-level aggregated loss
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch.

        Compute perplexity from aggregated loss (correct approach).
        """
        # Compute perplexity from aggregated loss (not from averaged perplexities!)
        avg_loss = self.train_loss.compute()
        train_perplexity = torch.exp(avg_loss)
        self.log("train/perplexity", train_perplexity, prog_bar=True)

        # Log learning rate (critical for debugging training issues!)
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", current_lr, prog_bar=False)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels
            batch_idx: Index of the current batch

        """
        loss, _logits = self.model_step(batch)

        # Update metrics (don't log perplexity here - it's incorrect!)
        self.val_loss(loss)

        # Log step-level loss (optional, for monitoring)
        self.log("val/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)

        # Log epoch-level aggregated loss
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch.

        Compute perplexity from aggregated loss (correct approach).
        """
        # Compute perplexity from aggregated loss
        avg_loss = self.val_loss.compute()
        val_perplexity = torch.exp(avg_loss)

        # Update best loss tracker
        self.val_loss_best(avg_loss)

        # Log metrics
        self.log("val/perplexity", val_perplexity, prog_bar=True)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels
            batch_idx: Index of the current batch

        """
        loss, _logits = self.model_step(batch)

        # Update metrics
        self.test_loss(loss)

        # Log epoch-level aggregated loss
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch.

        Compute perplexity from aggregated loss (correct approach).
        """
        # Compute perplexity from aggregated loss
        avg_loss = self.test_loss.compute()
        test_perplexity = torch.exp(avg_loss)
        self.log("test/perplexity", test_perplexity, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit, validate, test, or predict.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'

        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and optionally lr_scheduler configuration

        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            # Check if scheduler is ReduceLROnPlateau (requires special handling)
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val/loss",  # Metric to monitor
                "interval": "epoch",  # Check at end of epoch (or at val_check_interval)
                "frequency": 1,
            }

            # ReduceLROnPlateau requires the metric value
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler_config["strict"] = True  # Ensure metric is available
                scheduler_config["name"] = "ReduceLROnPlateau"

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # Test the LoRA module

    print("Testing SmolLM LoRA module initialization...")
    model = SmolLMLoRALitModule(
        model_name="HuggingFaceTB/SmolLM-135M",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        optimizer=None,  # Will be set by Lightning
        scheduler=None,
        compile=False,
        dropout=0.1,
    )
    print(f"Model loaded: {model.hparams.model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print("✓ SmolLM LoRA module initialized successfully!")
