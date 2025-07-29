# STL
from typing import Tuple
# 3rd Party
import torch
from lightning.pytorch.callbacks import Callback
import lightning as L
# Local

class TSEStatsCallback(Callback):

    def __init__(self):
        self.no_target = 0

        self.__summarized = False
        self.__tp = 0
        self.__pred_pos = 0
        self.__support = 0

        self.__fn_wrongtarg = 0
        self.__fn_wrongstance = 0
        self.__fp_wrongtarg = 0
        self.__fp_wrongstance = 0

    def reset(self):
        self.__summarized = False
        self.__tp = 0
        self.__pred_pos = 0
        self.__support = 0

        self.__fn_wrongtarg = 0
        self.__fn_wrongstance = 0
        self.__fp_wrongtarg = 0
        self.__fp_wrongstance = 0


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

        pred_pos = target_preds != self.no_target
        label_has_target = target_labels != self.no_target

        self.__pred_pos += int(torch.sum(pred_pos))

        pred_pos_inds = torch.where(pred_pos)
        self.__fp_wrongtarg += int(torch.sum(target_preds[pred_pos_inds] != target_labels[pred_pos_inds]))
        self.__fp_wrongstance += int(torch.sum(torch.logical_and(
            target_preds[pred_pos_inds] == target_labels[pred_pos_inds],
            stance_preds[pred_pos_inds] != stance_labels[pred_pos_inds]
        )))

        label_has_target_inds = torch.where(label_has_target)
        target_preds = target_preds[label_has_target_inds]
        stance_preds = stance_preds[label_has_target_inds]
        target_labels = target_labels[label_has_target_inds]
        stance_labels = stance_labels[label_has_target_inds]
        self.__support += target_labels.numel()
        self.__fn_wrongtarg   += int(torch.sum(target_preds != target_labels))
        self.__fn_wrongstance += int(torch.sum(torch.logical_and(target_preds == target_labels, stance_preds != stance_labels)))

        self.__tp += int(torch.sum(torch.logical_and(target_preds == target_labels, stance_preds == stance_labels)))

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
        results['fn_wrongtarg'] = self.__fn_wrongtarg
        results['fn_wrongstance'] = self.__fn_wrongstance
        results['fp_wrongtarg'] = self.__fp_wrongtarg
        results['fp_wrongstance'] = self.__fp_wrongstance
        results['pred_pos'] = self.__pred_pos
        results['support'] = self.__support
        results['tp'] = self.__tp

        _, _2, results['tse_f1'] = \
            TSEStatsCallback.compute_metrics(self.__tp, self.__pred_pos, self.__support)

        results = {f"{stage}_{k}":v for k,v in results.items()}
        for (k, v) in results.items():
            pl_module.log(k, v, on_step=False, on_epoch=True)
