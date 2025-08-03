# STL
from typing import Tuple
# 3rd Party
import torch
from lightning.pytorch.callbacks import Callback
import lightning as L
from torchmetrics.functional.classification import multiclass_stat_scores
# Local

class TargetClassificationStatsCallback(Callback):

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.__summarized = False
        self.__stats = self.__empty_stats()

    def __empty_stats(self):
        return torch.zeros((self.n_classes, 3), dtype=torch.long)

    def reset(self):
        self.__summarized = False
        self.__stats = self.__empty_stats()

    def record(self,
               target_preds: torch.Tensor,
               target_labels: torch.Tensor):
        if self.__summarized:
            raise ValueError("Must reset before recording more results")
        # Indices [0, 1, 3] correspond to [tp, fp, fn]
        rval = multiclass_stat_scores(target_preds, target_labels, self.n_classes, average='none')[:, [0, 1, 3]]
        self.__stats += rval.to(self.__stats.device)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset()
    def on_test_epoch_start(self, trainer, pl_module):
        self.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self.record(outputs.target_preds, batch['target'])

    def on_validation_epoch_end(self, trainer, pl_module):
        return self._on_epoch_end(trainer, pl_module, "val")
    def on_test_epoch_end(self, trainer, pl_module):
        return self._on_epoch_end(trainer, pl_module, "test")

    @staticmethod
    def compute_metrics(tp, fp, fn):
        pred_pos = tp + fp
        support = tp + fn
        prec = torch.where(pred_pos > 0, tp / pred_pos, 0.)
        rec = torch.where(support > 0, tp / support, 0.)
        denom = prec + rec
        f1 = torch.where(denom > 0, 2 * prec * rec / denom, 0.)
        return prec, rec, f1

    def _on_epoch_end(self, trainer, pl_module: L.LightningModule, stage):
        tp = self.__stats[:, 0]
        fp = self.__stats[:, 1]
        fn = self.__stats[:, 2]

        _, _2, class_f1 = TargetClassificationStatsCallback.compute_metrics(tp, fp, fn)
        macro_f1 = torch.mean(class_f1)

        agg_tp = torch.sum(tp)
        agg_fp = torch.sum(fp)
        agg_fn = torch.sum(fn)
        _, _2, micro_f1 = TargetClassificationStatsCallback.compute_metrics(agg_tp, agg_fp, agg_fn)

        results = {"macro_f1": macro_f1, "micro_f1": micro_f1}

        results = {f"{stage}_{k}":v for k,v in results.items()}
        for (k, v) in results.items():
            pl_module.log(k, v, on_step=False, on_epoch=True)
