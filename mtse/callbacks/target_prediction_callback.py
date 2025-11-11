import os
import json
import sys
import enum
import csv
from contextlib import contextmanager
from typing import Optional
from collections import defaultdict
import typing
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
from ..data.target_pred import TargetPred
from ..constants import ID_TO_LANG

@enum.unique
class TargetLevel(enum.IntEnum):
    none = 0
    generated = 1
    mapped = 2

def unique_consecutive(seq):
    if not seq:
        return
    yield seq[0]
    last = seq[0]
    for i in range(1, len(seq)):
        if seq[i] != last:
            yield seq[i]
            last = seq[i]

class TargetPredictionWriter(BasePredictionWriter, TargetMixin):
    def __init__(self,
                 out_dir: os.PathLike,
                 targets_path: os.PathLike,
                 embeddings_path: Optional[os.PathLike] = None,
                 target_level: TargetLevel = TargetLevel.mapped,
                 related_threshold: float = DEFAULT_RELATED_THRESHOLD,
                 ):
        BasePredictionWriter.__init__(self, write_interval='batch')
        TargetMixin.__init__(self, targets_path)
        self.out_dir = out_dir
        self.target_level = target_level

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
        self.__gen_targets_files = dict()
        self.__map_targets_files = dict()

        self.__gen_fieldnames = ["Sample", "Untranslated Target", "Generated Target", "GT Target", "Lang"]
        self.__map_fieldnames = ["Sample", "Untranslated Target", "Generated Target", "Mapped Target", "GT Target", "Lang"]


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

    def __get_gen_writer(self, source_path):
        label = os.path.basename(source_path)
        # the path in the filemap must be relative
        out_basename = f"{label}.target_gens.csv"
        self.__gen_targets_files[source_path] = out_basename

        return self.__get_writer(
            os.path.join(self.out_dir, out_basename),
            self.__gen_fieldnames,
            label,
            "target_gen"
        )

    def __get_map_writer(self, source_path):
        label = os.path.basename(source_path)
        out_basename = f"{label}.target_preds.csv"
        self.__map_targets_files[source_path] = out_basename

        return self.__get_writer(
            os.path.join(self.out_dir, out_basename),
            self.__map_fieldnames,
            label,
            "target_pred"
        )

    @staticmethod
    def __add_langs(rows, lang_strs):
        last_sid = None
        i = -1
        for row in rows:
            sid = row['Sample']
            if sid != last_sid:
                i += 1
                last_sid = sid
            row['Lang'] = lang_strs[i]

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if self.target_level <= TargetLevel.none:
            return

        source_paths = batch['source_path']
        assert all(p == source_paths[0] for p in source_paths)
        source_path = source_paths[0]

        target_labels = batch['target'].flatten().detach().cpu().tolist()
        str_labels = [self.targets[t] for t in target_labels]
        index_start = self.__sample_counter[source_path]

        gen_rows = None
        map_rows = None
        if getattr(prediction, "target_preds", None) is None:
            if hasattr(prediction, "generate_output"):
                prediction = prediction.generate_output
                assert isinstance(prediction, GenerateBeamEncoderDecoderOutput)

            if isinstance(prediction, GenerateBeamEncoderDecoderOutput):
                all_texts, sample_inds = detokenize_generated_targets(prediction, pl_module.tokenizer)
                zerobased_inds = torch.tensor(sample_inds, device=pl_module.device)
                sample_inds = [sind + index_start for sind in sample_inds]
            else:
                all_texts: List[str] = prediction.target_gens
                zerobased_inds = prediction.sample_inds - torch.min(prediction.sample_inds)
                sample_inds = prediction.sample_inds.detach().cpu().tolist()
            untrans_gens = batch.get('target_untrans', all_texts)
            gen_rows = [
                {"Sample": sind,
                 "Generated Target": text,
                 "Untranslated Target": ut_target,
                 "GT Target": str_labels[zind]
            } for zind, sind, text, ut_target in zip(zerobased_inds, sample_inds, all_texts, untrans_gens)]

            if self.target_level >= TargetLevel.mapped:
                if self.fast_text is None or self.__target_embeddings is None:
                    raise ValueError(f"You need to instantiate {self.__class__} with `embeddings_path` set")

                if self.__device is None:
                    self.__device = pl_module.device
                    self.__target_embeddings = self.__target_embeddings.to(self.__device)

                target_preds, all_text_inds = map_targets(self.fast_text,
                                                           self.__target_embeddings,
                                                           all_texts,
                                                           zerobased_inds,
                                                           self.related_threshold)
                # FIXME: Put the i < len(all_texts) handling logic in map_targets itself
                # This handles the case where segment_max_coo can't find a maximum...?
                freeform_preds = [all_texts[i] if i < len(all_texts) else "" for i in all_text_inds]
                untrans_preds = [untrans_gens[i] if i < len(all_texts) else "" for i in all_text_inds]

                sample_inds = unique_consecutive(sample_inds)
                map_rows = [{"Sample": sind,
                    "Generated Target": freeform_pred,
                    "Mapped Target": self.targets[target_pred],
                    "Untranslated Target": ut_pred,
                    "GT Target": str_labels[i]
                    } for i, (sind, freeform_pred, ut_pred, target_pred) in enumerate(zip(sample_inds, freeform_preds, untrans_preds, target_preds))
                ]
        else:
            if hasattr(prediction, "target_gens"):
                print(f"Warning: Skipping logging of target_gens for {source_path}", file=sys.stderr)
            if self.target_level < TargetLevel.mapped:
                return

            # If we have target_preds, the mapping has already been done
            # For simplicity, ignore any target_gen fields
            target_preds = [self.targets[p] for p in prediction.target_preds.flatten().detach().cpu().tolist()]
            untrans_preds = batch.get('target_untrans', target_preds)
            assert len(untrans_preds) == len(target_preds)

            if hasattr(prediction, "sample_inds"):
                sample_inds = prediction.sample_inds
            else:
                sample_inds = torch.arange(0, len(target_preds))
            sample_inds = unique_consecutive(sample_inds.detach().cpu().tolist())
            map_rows = [{
                "Sample": sind,
                "Generated Target": pred,
                "Mapped Target": pred,
                "Untranslated Target": ut_pred,
                "GT Target": str_labels[i]
                } for i, (sind, pred, ut_pred) in enumerate(zip(sample_inds, target_preds, untrans_preds))]

        
        lang_strs = batch['lang'] if 'lang' in batch else None

        if gen_rows is not None:
            if lang_strs is not None:
                TargetPredictionWriter.__add_langs(gen_rows, lang_strs)
            with self.__get_gen_writer(source_path) as writer:
                writer.writerows(gen_rows)
        if map_rows is not None:
            if lang_strs is not None:
                TargetPredictionWriter.__add_langs(map_rows, lang_strs)
            with self.__get_map_writer(source_path) as writer:
                writer.writerows(map_rows)
        self.__sample_counter[source_path] += len(target_labels)

    def _on_epoch_end(self, trainer, pl_module):
        with open(os.path.join(self.out_dir, "target_gen_map.json"), 'w') as w:
            json.dump(self.__gen_targets_files, w)
        with open(os.path.join(self.out_dir, "target_pred_map.json"), 'w') as w:
            json.dump(self.__map_targets_files, w)
    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module)
    def on_predict_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module)