# STL
import os
import csv
from contextlib import contextmanager
from collections import defaultdict
# 3rd Party
from lightning.pytorch.callbacks import BasePredictionWriter
# Local
from ..constants import C_SAMPLE, C_PRED_STANCE

class StancePredictionWriter(BasePredictionWriter):
    def __init__(self, out_dir: os.PathLike):
        super().__init__(write_interval='batch')
        self.out_dir = out_dir
        self.__started_file = set()
        self.__sample_counter = defaultdict(int)

    
    @staticmethod
    def __cons_writer(file_handle):
        return csv.DictWriter(file_handle, fieldnames=[C_SAMPLE, C_PRED_STANCE], lineterminator='\n')

    @contextmanager
    def __get_writer(self, source_path):
        label = os.path.basename(source_path)
        out_path = os.path.join(self.out_dir, f"{label}.stance_preds.csv")
        if label in self.__started_file:
            try:
                with open(out_path, 'a') as w:
                    yield self.__cons_writer(w)
            finally:
                pass
        else:
            self.__started_file.add(label)
            try:
                with open(out_path, 'w') as w:
                    writer = self.__cons_writer(w)
                    writer.writeheader()
                    yield writer
            finally:
                pass

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        source_paths = batch['source_path']
        assert all(p == source_paths[0] for p in source_paths)
        source_path = source_paths[0]
        index_start = self.__sample_counter[source_path]
        stance_preds = prediction.stance_preds.detach().cpu().tolist()

        stance_preds = [{C_SAMPLE: i, C_PRED_STANCE: pred} for i,pred in enumerate(stance_preds, start=index_start)]
        self.__sample_counter[source_path] += len(stance_preds)
        with self.__get_writer(source_path) as writer:
            writer.writerows(stance_preds)

