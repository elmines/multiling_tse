# STL
# 3rd Party
import torch
from lightning.pytorch.callbacks import Callback
import lightning as L
from torchmetrics.functional.classification import multiclass_stat_scores
# Local
from ..data.stance import *
from .utils import _compute_class_metrics


class StanceClassificationStatsCallback(Callback):

    def __init__(self, stance_type: StanceType = 'tri'):
        self.__stance_type = STANCE_TYPE_MAP[stance_type]

        enum_names = [s.name for s in self.__stance_type]
        self.__favor_index = enum_names.index('favor')
        self.__against_index = enum_names.index('against')


        self.__stats = self.__empty_stats()

    def __reset(self):
        self.__stats = self.__empty_stats()

    def __empty_stats(self):
        return torch.zeros((len(self.__stance_type), 3), dtype=torch.long)

    def __record(self,
               stance_preds: torch.Tensor,
               stance_labels: torch.Tensor):
        # Indices [0, 1, 3] correspond to [tp, fp, fn]
        batch_stats = multiclass_stat_scores(stance_preds, stance_labels, len(self.__stance_type), average='none')[:, [0, 1, 3]]
        self.__stats += batch_stats.to(self.__stats.device)

    def on_test_epoch_start(self, trainer, pl_module):
        self.__reset()
    def on_validation_epoch_start(self, trainer, pl_module):
        self.__reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'val')
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'test')
    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage):
        self.__record(outputs.stance_preds, batch['stance'])

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(pl_module, 'val')
    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(pl_module, 'test')
    def _on_epoch_end(self, pl_module: L.LightningModule, stage):
        _, _, micro_f1 = _compute_class_metrics(*torch.sum(self.__stats, dim=0))

        tp, fp, fn = self.__stats.transpose(1, 0)

        _, _, class_f1s = _compute_class_metrics(tp, fp, fn)
        macro_f1 = torch.mean(class_f1s)
        bimacro_f1 = torch.mean(class_f1s[ [self.__favor_index, self.__against_index] ])

        results = {
            "stance/micro_f1": micro_f1,
            "stance/bimacro_f1": bimacro_f1,
            "stance/macro_f1": macro_f1
        }

        results = {f"{stage}/{k}":v for k,v in results.items()}
        for (k, v) in results.items():
            pl_module.log(k, v, on_step=False, on_epoch=True, add_dataloader_idx=False)
