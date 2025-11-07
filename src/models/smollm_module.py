from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric
from transformers import AutoModelForCausalLM


class SmolLMLitModule(LightningModule):
    """LightningModule for SmolLM-135M language model fine-tuning.

    This module wraps HuggingFace's SmolLM-135M model for causal language modeling.
    The model is trained to predict the next token in a sequence.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None,
        compile: bool,
    ) -> None:
        """Initialize a `SmolLMLitModule`.

        :param model_name: The HuggingFace model name/path (e.g., "HuggingFaceTB/SmolLM-135M").
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model with torch.compile (PyTorch 2.0+).
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Load pretrained model from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best validation loss
        self.val_loss_best = MeanMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass through the model.

        :param input_ids: Token IDs tensor of shape [batch_size, seq_len].
        :param attention_mask: Attention mask tensor of shape [batch_size, seq_len].
        :param labels: Labels tensor for loss calculation (same as input_ids for causal LM).
        :return: Dictionary containing loss and logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing input_ids, attention_mask, and labels.
        :return: A tuple containing (loss, logits).
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
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing input_ids, attention_mask, and labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of loss between model predictions and targets.
        """
        loss, _logits = self.model_step(batch)

        # Calculate perplexity (exp of loss)
        perplexity = torch.exp(loss)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing input_ids, attention_mask, and labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _logits = self.model_step(batch)

        # Calculate perplexity
        perplexity = torch.exp(loss)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_loss_best` as a value through `.compute()` method
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing input_ids, attention_mask, and labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _logits = self.model_step(batch)

        # Calculate perplexity
        perplexity = torch.exp(loss)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.

        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # Simple test
    print("Testing SmolLM module initialization...")
    model = SmolLMLitModule(
        model_name="HuggingFaceTB/SmolLM-135M",
        optimizer=torch.optim.AdamW,
        scheduler=None,
        compile=False,
    )
    print(f"Model loaded: {model.hparams.model_name}")
    print(f"Model type: {type(model.model)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ SmolLM module initialized successfully!")
