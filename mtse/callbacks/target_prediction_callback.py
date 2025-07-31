import os

from lightning.pytorch.callbacks import BasePredictionWriter

from ..constants import UNRELATED_TARGET

class TargetPredictionWriter(BasePredictionWriter):
    def __init__(self, out_dir: os.PathLike):
        super().__init__(write_interval='batch')
        self.out_dir = out_dir
        self.out_path = os.path.join(self.out_dir, 'target_preds.txt')
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        target_preds = prediction.target_preds.flatten().detach().cpu().tolist()
        target_lookup = [UNRELATED_TARGET] + pl_module.targets
        str_preds = [target_lookup[t] for t in target_preds]
        with open(self.out_path, 'a', encoding='utf-8') as w:
            w.write('\n'.join(str_preds) + '\n')