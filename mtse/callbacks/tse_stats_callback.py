# STL
import csv
from typing import Tuple, Dict, Optional
import pathlib
# 3rd Party
import torch
from torchmetrics.functional.classification import multiclass_stat_scores
from lightning.pytorch.callbacks import Callback
import lightning as L
# Local

class TSEStatsCallback(Callback):

    def __init__(self):
        self.no_target = 0

        self.__summarized = False
        self.__tp = 0
        self.__all_pos = 0
        self.__support = 0


    @staticmethod
    def compute_metrics(tp, pred_pos, support) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute precision, recall, and f1
        """
        precision = tp / pred_pos if pred_pos > 0 else 0
        recall = tp / support if support > 0 else 0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0
        return precision, recall, f1


    def record(self,
               target_preds: torch.Tensor,
               stance_preds: torch.Tensor,
               target_labels: torch.Tensor,
               stance_labels: torch.Tensor):
        if self.__summarized:
            raise ValueError("Must reset F1Calc before recording more results")

        self.__all_pos += int(torch.sum(target_preds != self.no_target))
        self.__support += int(torch.sum(target_labels != self.no_target))

        label_has_target = torch.where(target_labels != self.no_target)
        target_preds = target_preds[label_has_target]
        stance_preds = stance_preds[label_has_target]
        target_labels = target_labels[label_has_target]
        stance_labels = stance_labels[label_has_target]
        self.__tp += int(torch.logical_and(target_preds == stance_preds, target_labels == stance_labels))

    def reset(self):
        self.__summarized = False
        self.__tp = 0
        self.__all_pos = 0
        self.__support = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset()
    def on_test_epoch_start(self, trainer, pl_module):
        self.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self.record(outputs.target_preds, outputs.stance_preds, batch['target'], batch['stance'])

    def on_validation_epoch_end(self, trainer, pl_module):
        return self._on_epoch_end(trainer, pl_module, "val")
    def on_test_epoch_end(self, trainer, pl_module):
        return self._on_epoch_end(trainer, pl_module, "test")
    def _on_epoch_end(self, trainer, pl_module: L.LightningModule, stage):
        results = {}
        _, _2, results['tse_f1'] = TSEStatsCallback.compute_metrics(self.__tp, self.__all_pos, self.__support)
        for (k, v) in filter(lambda pair: pair[0].endswith('f1'), results.items()):
            pl_module.log(k, v, on_step=False, on_epoch=True)
