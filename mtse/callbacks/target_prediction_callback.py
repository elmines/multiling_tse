import os
import csv
from contextlib import contextmanager

from lightning.pytorch.callbacks import BasePredictionWriter

from ..constants import UNRELATED_TARGET

class TargetPredictionWriter(BasePredictionWriter):
    def __init__(self, out_dir: os.PathLike):
        super().__init__(write_interval='batch')
        self.out_dir = out_dir

        self.__started_file = set()
        self.__fieldnames = ["Mapped Target", "GT Target"]

    def __cons_writer(self, file_handle):
        return csv.DictWriter(file_handle, fieldnames=self.__fieldnames, lineterminator='\n')

    @contextmanager
    def __get_writer(self, dataloader_idx):
        out_path = os.path.join(self.out_dir, f"target_preds.{dataloader_idx}.txt")
        if dataloader_idx in self.__started_file:
            try:
                with open(out_path, 'a') as w:
                    yield self.__cons_writer(w)
            finally:
                pass
        else:
            self.__started_file.add(dataloader_idx)
            try:
                with open(out_path, 'w') as w:
                    writer = self.__cons_writer(w)
                    writer.writeheader()
                    yield writer
            finally:
                pass

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        target_preds = prediction.target_preds.flatten().detach().cpu().tolist()
        target_labels = batch['target'].flatten().detach().cpu().tolist()
        target_lookup = pl_module.targets
        str_preds = [target_lookup[t] for t in target_preds]
        str_labels = [target_lookup[t] for t in target_labels]
        row_dicts = [{"Mapped Target": pred, "GT Target": label} for pred, label in zip(str_preds, str_labels)]
        with self.__get_writer(dataloader_idx) as writer:
            writer.writerows(row_dicts)
