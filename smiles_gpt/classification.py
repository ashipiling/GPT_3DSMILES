"""HuggingFace-compatible classification and regression models including
pytorch-lightning models.
"""

__all__ = ("BypassNet", "ClassificationHead", "ClassifierLitModel",
           "GPT2ForSequenceClassification", "RegressorLitModel",
           "SequenceClassifierOutput")

from dataclasses import dataclass
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, AveragePrecision
from transformers import AdamW, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.adapters.model_mixin import ModelWithHeadsAdaptersMixin


@dataclass
class SequenceClassifierOutput(SequenceClassifierOutputWithPast):
    target: Optional[torch.LongTensor] = None


class GPT2ForSequenceClassification(ModelWithHeadsAdaptersMixin, GPT2PreTrainedModel):
    """HuggingFace-compatible single- and multi-output (-task) classification model.
    `config` must be a `GPT2Config` instance with additional `num_tasks` and `num_labels`
    properties. For multi-task classification, the output is Bypass network with the
    reduction factor = `config.n_embd // config.n_head`.
    """

    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight", r"output\..*"]

    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = config.num_tasks
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)

        if self.num_tasks > 1:
            self.output = BypassNet(
                config.n_embd, config.n_embd // config.n_head,
                config.num_tasks, config.num_labels,
                config.embd_pdrop)
        else:
            self.output = ClassificationHead(
                config.n_embd, config.n_embd // config.n_head,
                config.num_labels, config.embd_pdrop)

        self.init_weights()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, adapter_names=None,
                label_mask=None):
        return_dict = return_dict or self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
            adapter_names=adapter_names)

        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert self.config.pad_token_id is not None or batch_size == 1, \
            "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(
                    input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1

        if self.num_tasks == 1:
            logits = self.output(hidden_states)[range(batch_size), sequence_lengths]
        else:
            logits = self.output(hidden_states, batch_size, sequence_lengths)

        loss = None
        if labels is not None:
            if self.num_labels == 2:
                if label_mask is not None:
                    nonempty_tasks = (label_mask == 1).view(-1)
                    nonempty_logits = logits.view(-1, self.num_labels)[nonempty_tasks, :]
                    nonempty_labels = labels.view(-1)[nonempty_tasks]
                else:
                    nonempty_logits = logits.view(-1, self.num_labels)
                    nonempty_labels = labels.view(-1)

                if len(labels.size()) == 1:
                    labels = labels.reshape(1, -1)

                loss = F.cross_entropy(nonempty_logits, nonempty_labels)
            elif self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                raise NotImplementedError(
                    "Only binary classification and regression supported.")

        if self.num_tasks > 1:
            logits = logits.transpose(1, 2)

        if labels is not None and self.num_labels == 2 and self.num_tasks == 1:
            if label_mask is not None:
                labels = labels.view(-1)
            else:
                labels = nonempty_labels

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, target=labels,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions)


class BypassNet(nn.Module):
    """Bypass multi-task network from MoleculeNet project [Wu et al., 2018].
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_tasks: int, num_labels: int = 2,
                 dropout: float = 0.2, use_bias: bool = False):
        super().__init__()
        self.independent = nn.ModuleList([
            ClassificationHead(hidden_size, intermediate_size,
                               num_labels, dropout, use_bias)
            for _ in range(num_tasks)])
        self.shared = ClassificationHead(hidden_size, intermediate_size,
                                         num_labels, dropout, use_bias)

    def forward(self, hidden_states, batch_size, sequence_lengths):
        logits_list: List[torch.Tensor] = []
        for layer in self.independent:
            logits_list.append(layer(hidden_states))
        shared_logits: torch.Tensor = self.shared(hidden_states)
        for i in range(len(logits_list)):
            logits_list[i] = (logits_list[i] + shared_logits)[range(batch_size),
                                                              sequence_lengths]
        return torch.stack(logits_list, dim=1)


class ClassificationHead(nn.Module):
    """Two-layer feed-forward network with GELU activation and intermediate dropout.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_labels: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(intermediate_size, num_labels, bias=use_bias)

    def forward(self, x, *args, **kwargs):
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.out_proj(x)


class ClassifierLitModel(pl.LightningModule):
    """Pytorch-lightning module for single- or multi-task classification. Trains GPT2
    model using `AdamW` optimizer with exponential LR scheduler. Evaluates valid and
    test data on AUC-ROC and AUC-PRC.

    Args:
        transformer (`GPT2Model`): (Pretrained) HuggingFace GPT2 model.
        num_tasks (int): The number of classification tasks.
        has_empty_labels (bool)
        batch_size (int)
        learning_rate (float)
        scheduler_lambda (float)
        scheduler_step (int)
        weight_decay (float)
    """

    def __init__(self, transformer: GPT2Model, num_tasks: int, has_empty_labels: bool,
                 batch_size: int, learning_rate: float, scheduler_lambda: float,
                 scheduler_step: int, weight_decay: float, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=("transformer", "num_tasks", "has_empty_labels"))
        self.transformer = transformer
        self.num_tasks = num_tasks

        def get_metrics(metric_cls):
            return [metric_cls(num_classes=2) for _ in range(num_tasks)]

        if has_empty_labels:
            self.train_roc = get_metrics(AUROC)
            self.val_roc = get_metrics(AUROC)
            self.test_roc = get_metrics(AUROC)

            self.train_prc = get_metrics(AveragePrecision)
            self.val_prc = get_metrics(AveragePrecision)
            self.test_prc = get_metrics(AveragePrecision)

            self.step = self._step_empty
            self.epoch_end = self._epoch_end_empty
        else:
            self.train_roc = AUROC(num_classes=2)
            self.val_roc = AUROC(num_classes=2)
            self.test_roc = AUROC(num_classes=2)

            self.train_prc = AveragePrecision(num_classes=2)
            self.val_prc = AveragePrecision(num_classes=2)
            self.test_prc = AveragePrecision(num_classes=2)

            self.step = self._step_nonempty
            self.epoch_end = self._epoch_end_nonempty

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def _step_empty(self, batch, batch_idx, roc, prc):
        outputs = self(**batch)

        if self.num_tasks == 1:
            outputs["target"] = outputs["target"][:, None]
            outputs["logits"] = outputs["logits"][:, :, None]

        for task_id in range(self.num_tasks):
            target = outputs["target"][:, task_id]
            nonempty_entries = target != -1
            target = target[nonempty_entries]

            if target.unique().size(0) > 1:
                logits = outputs["logits"][:, :, task_id][nonempty_entries]

                roc[task_id](logits, target)
                prc[task_id](logits, target)

        return {"loss": outputs["loss"]}

    def _step_nonempty(self, batch, batch_idx, roc, prc):
        outputs = self(**batch)

        logits, target = outputs["logits"], outputs["target"]
        if target.unique().size(0) > 1:
            roc(logits, target)
            prc(logits, target)

        return {"loss": outputs["loss"]}

    def _epoch_end_empty(self, outputs_ignored, roc, prc, prefix):
        mean_roc = sum(a.compute() for a in roc) / self.num_tasks
        self.log(f"{prefix}_roc", mean_roc, on_step=False, on_epoch=True, prog_bar=True)
        mean_prc = sum(p.compute()[1] for p in prc) / self.num_tasks
        self.log(f"{prefix}_prc", mean_prc, on_step=False, on_epoch=True, prog_bar=True)

    def _epoch_end_nonempty(self, outputs, roc, prc, prefix):
        self.log(f"{prefix}_roc", roc.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_prc", prc.compute()[1],
                 on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.train_roc, self.train_prc)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, self.train_roc, self.train_prc, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.val_roc, self.val_prc)

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, self.val_roc, self.val_prc, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, self.test_roc, self.test_prc)

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, self.test_roc, self.test_prc, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.scheduler_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler,
                                 "interval": "step",
                                 "frequency": self.hparams.scheduler_step}}


class RegressorLitModel(pl.LightningModule):
    def __init__(self, transformer: GPT2Model,
                 batch_size: int, learning_rate: float, scheduler_lambda: float,
                 scheduler_step: int, weight_decay: float, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore="transformer")
        self.transformer = transformer

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def step(self, batch, batch_idx):
        outputs = self(**batch)
        rmse_loss = torch.sqrt(outputs["loss"])
        return {"loss": rmse_loss}

    def epoch_end(self, outputs, prefix):
        mean_rmse = torch.mean(torch.tensor([out["loss"] for out in outputs]))
        self.log(f"{prefix}_rmse", mean_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.scheduler_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler,
                                 "interval": "step",
                                 "frequency": self.hparams.scheduler_step}}
