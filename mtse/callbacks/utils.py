import torch

def _compute_class_metrics(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    pred_pos = tp + fp
    support = tp + fn
    prec = torch.where(pred_pos > 0, tp / pred_pos, 0.)
    rec = torch.where(support > 0, tp / support, 0.)
    denom = prec + rec
    f1 = torch.where(denom > 0, 2 * prec * rec / denom, 0.)
    return prec, rec, f1

