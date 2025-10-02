# STL
from typing import Tuple
import dataclasses
from collections import defaultdict
from typing import Optional, List
import functools
# 3rd Party
import torch
from lightning.pytorch.callbacks import Callback
import lightning as L
# Local

class TSEStatsCallback(Callback):

    @dataclasses.dataclass
    class CorpStats:
        tp: int = 0
        pred_pos: int = 0
        support: int = 0
        fn_wrongtarg: int = 0
        fn_wrongstance: int = 0
        fp_wrongtarg: int = 0
        fp_wrongstance: int = 0
        correct: int = 0
        total: int = 0

        def __add__(self, rhs):
            return TSEStatsCallback.CorpStats(
                tp=self.tp + rhs.tp,
                pred_pos=self.pred_pos + rhs.pred_pos,
                support=self.support + rhs.support,
                fn_wrongtarg=self.fn_wrongtarg + rhs.fn_wrongtarg,
                fn_wrongstance=self.fn_wrongstance + rhs.fn_wrongstance,
                fp_wrongtarg=self.fp_wrongtarg + rhs.fp_wrongtarg,
                fp_wrongstance=self.fp_wrongstance + rhs.fp_wrongstance,
                correct=self.correct + rhs.correct,
                total=self.total + rhs.total,
            )

    def __init__(self, full_metrics=False):
        self.no_target = 0
        self.full_metrics = full_metrics
        self.dataloader_labels = []
        self.__stats_by_corp = defaultdict(TSEStatsCallback.CorpStats)


    def reset(self):
        self.__stats_by_corp = defaultdict(TSEStatsCallback.CorpStats)

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
               stance_labels: torch.Tensor,
               dataloader_idx: int):
        corp_stats = self.__stats_by_corp[dataloader_idx]

        corp_stats.correct += int(torch.sum(torch.logical_or(
            torch.logical_and(target_preds == self.no_target, target_labels == self.no_target),
            torch.logical_and(target_preds == target_labels, stance_preds == stance_labels)
        )))
        corp_stats.total += stance_labels.numel()


        pred_pos = target_preds != self.no_target
        label_has_target = target_labels != self.no_target

        corp_stats.pred_pos += int(torch.sum(pred_pos))

        pred_pos_inds = torch.where(pred_pos)
        corp_stats.fp_wrongtarg += int(torch.sum(target_preds[pred_pos_inds] != target_labels[pred_pos_inds]))
        corp_stats.fp_wrongstance += int(torch.sum(torch.logical_and(
            target_preds[pred_pos_inds] == target_labels[pred_pos_inds],
            stance_preds[pred_pos_inds] != stance_labels[pred_pos_inds]
        )))

        label_has_target_inds = torch.where(label_has_target)
        target_preds = target_preds[label_has_target_inds]
        stance_preds = stance_preds[label_has_target_inds]
        target_labels = target_labels[label_has_target_inds]
        stance_labels = stance_labels[label_has_target_inds]
        corp_stats.support += target_labels.numel()
        corp_stats.fn_wrongtarg   += int(torch.sum(target_preds != target_labels))
        corp_stats.fn_wrongstance += int(torch.sum(torch.logical_and(target_preds == target_labels, stance_preds != stance_labels)))

        corp_stats.tp += int(torch.sum(torch.logical_and(target_preds == target_labels, stance_preds == stance_labels)))

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset()
    def on_test_epoch_start(self, trainer, pl_module):
        self.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self.record(outputs.target_preds, outputs.stance_preds, batch['target'], batch['stance'], dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        return self._on_epoch_end(trainer, pl_module, "val")
    def on_test_epoch_end(self, trainer, pl_module):
        return self._on_epoch_end(trainer, pl_module, "test")
    def _on_epoch_end(self, trainer, pl_module: L.LightningModule, stage):

        def log_stats(stats: TSEStatsCallback.CorpStats, dataloader_idx: Optional[int] = None):
            results = {}
            ldr_suffix = ""
            if dataloader_idx is not None:
                if dataloader_idx < len(self.dataloader_labels):
                    ldr_suffix = f"/{self.dataloader_labels[dataloader_idx]}"
                else:
                    ldr_suffix = f"/{dataloader_idx}"
            
            if self.full_metrics:
                results['tse/fn_wrongtarg'] = stats.fn_wrongtarg
                results['tse/fn_wrongstance'] = stats.fn_wrongstance
                results['tse/fp_wrongtarg'] = stats.fp_wrongtarg
                results['tse/fp_wrongstance'] = stats.fp_wrongstance
                results['tse/pred_pos'] = stats.pred_pos
                results['tse/support'] = stats.support
                results['tse/tp'] = stats.tp

                _, _2, results['tse/f1'] = \
                    TSEStatsCallback.compute_metrics(stats.tp, stats.pred_pos, stats.support)
                results['tse/acc'] = stats.correct / stats.total if stats.total > 0 else 0.0
                results['tse/nsamples'] = stats.total

                results = {f"{stage}/{k}{ldr_suffix}":v for k,v in results.items()}
                for (k, v) in results.items():
                    pl_module.log(k, v, on_step=False, on_epoch=True)
        agg_stats = functools.reduce(lambda accum,el: accum + el, self.__stats_by_corp.values())
        log_stats(agg_stats)
        if len(self.__stats_by_corp) > 1:
            for dataloader_idx, stats in self.__stats_by_corp.items():
                log_stats(stats, dataloader_idx)