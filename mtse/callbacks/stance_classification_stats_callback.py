# STL
from collections import defaultdict
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

        self.__stats_by_corp = defaultdict(lambda: self.__empty_stats())

    def __reset(self):
        self.__stats_by_corp = defaultdict(lambda: self.__empty_stats())

    def __empty_stats(self):
        return torch.zeros((len(self.__stance_type), 3), dtype=torch.long)

    def __record(self,
               stance_preds: torch.Tensor,
               stance_labels: torch.Tensor,
               dataloader_idx: int):
        corp_stats = self.__stats_by_corp[dataloader_idx]
        # Indices [0, 1, 3] correspond to [tp, fp, fn]
        batch_stats = multiclass_stat_scores(stance_preds, stance_labels, len(self.__stance_type), average='none')[:, [0, 1, 3]]
        corp_stats += batch_stats.to(corp_stats.device)

    def on_test_epoch_start(self, trainer, pl_module):
        self.__reset()
    def on_validation_epoch_start(self, trainer, pl_module):
        self.__reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'val')
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'test')
    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage):
        self.__record(outputs.stance_preds, batch['stance'], dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(pl_module, 'val')
    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(pl_module, 'test')
    def _on_epoch_end(self, pl_module: L.LightningModule, stage):
        stats_by_corp = dict(self.__stats_by_corp)
        assert stats_by_corp
        per_corpus_metrics = len(stats_by_corp) > 1
        results = dict()

        cum_tp = 0
        cum_fp = 0
        cum_fn = 0
        corpus_f1s = []
        for didx, stats in stats_by_corp.items():
            tp, fp, fn = stats.transpose(1, 0)
            cum_tp += tp
            cum_fp += fp
            cum_fn += fn
            if per_corpus_metrics:
                _, _, class_f1s = _compute_class_metrics(tp, fp, fn)
                corpus_f1 = torch.mean(class_f1s[[self.__favor_index, self.__against_index]])
                corpus_f1s.append(corpus_f1)
                results[f'bimacro_f1/{didx}'] = corpus_f1
        if per_corpus_metrics:
            results['avg_of_dataset_bimacro_f1'] = sum(corpus_f1s) / len(corpus_f1s)

        _, _, cum_class_f1s = _compute_class_metrics(cum_tp, cum_fp, cum_fn)
        results['bimacro_f1'] = torch.mean(cum_class_f1s[ [self.__favor_index, self.__against_index] ])

        results = {f"{stage}/stance/{k}":v for k,v in results.items()}
        for (k, v) in results.items():
            pl_module.log(k, v, on_step=False, on_epoch=True, add_dataloader_idx=False)
