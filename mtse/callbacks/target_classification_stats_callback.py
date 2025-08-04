# STL
from collections import defaultdict
# 3rd Party
import torch
from lightning.pytorch.callbacks import Callback
import lightning as L
from torchmetrics.functional.classification import multiclass_stat_scores
# Local

def _compute_corpus_metrics(tp, fp, fn):
    _, _2, class_f1 = _compute_class_metrics(tp, fp, fn)
    macro_f1 = torch.mean(class_f1)
    agg_tp = torch.sum(tp)
    agg_fp = torch.sum(fp)
    agg_fn = torch.sum(fn)
    _, _2, micro_f1 = _compute_class_metrics(agg_tp, agg_fp, agg_fn)
    return macro_f1, micro_f1


def _compute_class_metrics(tp, fp, fn):
    pred_pos = tp + fp
    support = tp + fn
    prec = torch.where(pred_pos > 0, tp / pred_pos, 0.)
    rec = torch.where(support > 0, tp / support, 0.)
    denom = prec + rec
    f1 = torch.where(denom > 0, 2 * prec * rec / denom, 0.)
    return prec, rec, f1


class TargetClassificationStatsCallback(Callback):

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.__stats_by_corp = defaultdict(lambda: self.__empty_stats())

    def __reset(self):
        self.__stats_by_corp = defaultdict(lambda: self.__empty_stats())

    def __empty_stats(self):
        return torch.zeros((self.n_classes, 3), dtype=torch.long)

    def record(self,
               target_preds: torch.Tensor,
               target_labels: torch.Tensor,
               dataloader_idx: int):
        corp_stats = self.__stats_by_corp[dataloader_idx]
        # Indices [0, 1, 3] correspond to [tp, fp, fn]
        batch_stats = multiclass_stat_scores(target_preds, target_labels, self.n_classes, average='none')[:, [0, 1, 3]]
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
        self.record(outputs.target_preds, batch['target'], dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(pl_module, 'val')
    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(pl_module, 'test')
    def _on_epoch_end(self, pl_module: L.LightningModule, stage):
        stats_by_corp = dict(self.__stats_by_corp)
        assert stats_by_corp
        results = dict()
        if len(stats_by_corp) == 1:
            results['macro_f1'], results['micro_f1'] = _compute_corpus_metrics(*stats_by_corp[0].transpose(1, 0))
        else:
            for (dataloader_idx, corp_stats) in stats_by_corp.items():
                macro_f1, micro_f1 = _compute_corpus_metrics(*corp_stats.transpose(1, 0))
                results[f'macro_f1/{dataloader_idx}'] = macro_f1
                results[f'micro_f1/{dataloader_idx}'] = micro_f1
            global_stats = sum(stats_by_corp.values())
            results['macro_f1'], results['micro_f1'] = _compute_corpus_metrics(*global_stats.transpose(1, 0))
        results = {f"{stage}_{k}":v for k,v in results.items()}
        for (k, v) in results.items():
            pl_module.log(k, v, on_step=False, on_epoch=True, add_dataloader_idx=False)
