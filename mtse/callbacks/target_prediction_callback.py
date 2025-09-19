import os
import enum
import csv
from contextlib import contextmanager
from typing import Optional
from collections import defaultdict
from typing import List
# 3rd Party
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from gensim.models import FastText
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
# Local
from ..modules.mixins import TargetMixin
from ..constants import DEFAULT_RELATED_THRESHOLD
from ..mapping import make_target_embeddings, detokenize_generated_targets, map_targets

@enum.unique
class TargetLevel(enum.IntEnum):
    none = 0
    generated = 1
    mapped = 2


class TargetPredictionWriter(BasePredictionWriter, TargetMixin):
    def __init__(self,
                 out_dir: os.PathLike,
                 targets_path: os.PathLike,
                 embeddings_path: Optional[os.PathLike] = None,
                 target_level: TargetLevel = TargetLevel.mapped,
                 related_threshold: float = DEFAULT_RELATED_THRESHOLD,
                 dataloader_labels: Optional[List[str]] = None
                 ):
        BasePredictionWriter.__init__(self, write_interval='batch')
        TargetMixin.__init__(self, targets_path)
        self.out_dir = out_dir
        self.target_level = target_level
        self.dataloader_labels = dataloader_labels or []

        self.related_threshold = related_threshold
        if embeddings_path is not None:
            self.fast_text = FastText.load(str(embeddings_path))
            self.__target_embeddings = torch.tensor(make_target_embeddings(self.targets, self.fast_text), device='cpu')
        else:
            self.fast_text = None
            self.__target_embeddings = None

        self.__device: Optional[str] = None

        self.__started_file = set()
        self.__sample_counter = defaultdict(int)

        self.__gen_fieldnames = ["Sample", "Generated Target", "GT Target"]
        self.__map_fieldnames = ["Sample", "Generated Target", "Mapped Target", "GT Target"]


    @staticmethod
    def __cons_writer(file_handle, fieldnames):
        return csv.DictWriter(file_handle, fieldnames=fieldnames, lineterminator='\n')

    @contextmanager
    def __get_writer(self, out_path, fieldnames, dataloader_idx, task):
        k = (dataloader_idx, task)
        if k in self.__started_file:
            try:
                with open(out_path, 'a') as w:
                    yield self.__cons_writer(w, fieldnames)
            finally:
                pass
        else:
            self.__started_file.add(k)
            try:
                with open(out_path, 'w') as w:
                    writer = self.__cons_writer(w, fieldnames)
                    writer.writeheader()
                    yield writer
            finally:
                pass

    def __get_gen_writer(self, dataloader_idx):
        label = self.dataloader_labels[dataloader_idx] if dataloader_idx < len(self.dataloader_labels) else dataloader_idx
        return self.__get_writer(
            os.path.join(self.out_dir, f"target_gens.{label}.txt"),
            self.__gen_fieldnames,
            dataloader_idx,
            "target_gen"
        )

    def __get_map_writer(self, dataloader_idx):
        label = self.dataloader_labels[dataloader_idx] if dataloader_idx < len(self.dataloader_labels) else dataloader_idx
        return self.__get_writer(
            os.path.join(self.out_dir, f"target_preds.{label}.txt"),
            self.__map_fieldnames,
            dataloader_idx,
            "target_pred"
        )

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if self.target_level <= TargetLevel.none:
            return

        target_labels = batch['target'].flatten().detach().cpu().tolist()
        str_labels = [self.targets[t] for t in target_labels]
        index_start = self.__sample_counter[dataloader_idx]

        gen_rows = None
        map_rows = None
        if isinstance(prediction, GenerateBeamEncoderDecoderOutput):
            all_texts, sample_inds = detokenize_generated_targets(prediction, pl_module.tokenizer)
            gen_rows = [{"Sample": index_start + i, "Generated Target": text, "GT Target": str_labels[i]} for i,text in zip(sample_inds, all_texts) ]

            if self.target_level >= TargetLevel.mapped:
                if self.fast_text is None or self.__target_embeddings is None:
                    raise ValueError(f"You need to instantiate {self.__class__} with `embeddings_path` set")

                if self.__device is None:
                    self.__device = pl_module.device
                    self.__target_embeddings = self.__target_embeddings.to(self.__device)

                target_preds, freeform_preds = map_targets(self.fast_text,
                                                           self.__target_embeddings,
                                                           all_texts,
                                                           sample_inds,
                                                           self.related_threshold)
                map_rows = [{"Sample": i + index_start,
                    "Generated Target": freeform_pred,
                    "Mapped Target": self.targets[target_pred],
                    "GT Target": str_labels[i]
                    } for i, (freeform_pred, target_pred) in enumerate(zip(freeform_preds, target_preds))
                ]
        else:
            assert hasattr(prediction, "target_preds")
            target_preds = prediction.target_preds.flatten().detach().cpu().tolist()
            # Pretend like we "generated" the targets, even though we actually only classified them
            gen_rows = [{"Sample": i + index_start,
                "Generated Target": pred,
                "GT Target": str_labels[i]} for i, pred in enumerate(str_labels)
            ]
            if self.target_level >= TargetLevel.mapped:
                # For TC, just copypaste the 
                map_rows = [{"Mapped Target": row["Generated Target"], **row} for row in gen_rows]
        if gen_rows is not None:
            with self.__get_gen_writer(dataloader_idx) as writer:
                writer.writerows(gen_rows)
        if map_rows is not None:
            with self.__get_map_writer(dataloader_idx) as writer:
                writer.writerows(map_rows)
        self.__sample_counter[dataloader_idx] += len(target_labels)