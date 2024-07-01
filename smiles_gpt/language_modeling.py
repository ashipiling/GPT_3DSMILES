"""Pytorch-lightning module for causal language modeling.
"""

__all__ = ("GPT2LitModel",)

import pytorch_lightning as pl
import torch


class GPT2LitModel(pl.LightningModule):
    """Lightning module for autoregressive (causal) transformer language modeling.
    Successfully tested on HuggingFace `GPT2LMHeadModel`.
    """

    def __init__(self, transformer, batch_size: int, learning_rate: float,
                 final_learning_rate: float, weight_decay: float, adam_eps: float,
                 adam_betas: tuple, scheduler_T_max: int,
                 save_model_every: int = 10_000, checkpoint: str = ""):
        super().__init__()
        self.save_hyperparameters(ignore=("transformer", "save_model_every",
                                          "checkpoints"))
        self.transformer = transformer
        self.save_model_every = save_model_every
        self.checkpoint = checkpoint or "./gpt2litmodel-logs"

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        if self.save_model_every > 0 and batch_idx % self.save_model_every == 0:
            self.transformer.save_pretrained(self.checkpoint)

        return {'loss': outputs['loss']}

    def training_epoch_end(self, outputs):
        if self.save_model_every > 0:
            self.transformer.save_pretrained(self.checkpoint)

        losses = [step_output["loss"] for step_output in outputs]
        mean_loss = torch.tensor(losses).mean()
        ppl = torch.exp(mean_loss)

        self.log("ppl", ppl, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        parameters = self.named_parameters()
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {"params": [p for n, p in parameters
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in parameters
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        optimizer = torch.optim.Adam(
            grouped_parameters, lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps, betas=self.hparams.adam_betas)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.hparams.scheduler_T_max,
            eta_min=self.hparams.final_learning_rate)

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'interval': 'step', 'frequency': 1}}
