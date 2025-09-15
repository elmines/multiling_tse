import os
import enum
import csv
from contextlib import contextmanager
from typing import List, Optional, Tuple
# 3rd Party
import torch
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from gensim.models import FastText
from transformers import PreTrainedTokenizerFast
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from torch_scatter import segment_max_coo
# Local
from ..modules.mixins import TargetMixin
from ..constants import TARGET_DELIMITER, UNRELATED_TARGET, DEFAULT_RELATED_THRESHOLD
from .utils import detokenize_generated_targets

@enum.unique
class TargetLevel(enum.Enum):
    NONE = 'none'
    GENERATED = 'generated'
    MAPPED = 'mapped'

def make_target_embeddings(targets: List[str], fast_text: FastText) -> np.ndarray:
    unstacked = []
    for target in targets:
        if target == UNRELATED_TARGET:
            # Will never match with the unrelated target, because no vector X can have
            # a nonzero cosine similarity with the zero vector
            unstacked.append(np.zeros([fast_text.vector_size], dtype=np.float32))
        else:
            vec = fast_text.wv[target.lower()].copy()
            # Normalize magnitude to unity--makes computing explicit cosine similarity unnecessary
            # Just find the target vector that has the highest dot product with the predicted target
            vec = vec / np.linalg.norm(vec) 
            unstacked.append(vec)
    # (n_targets, embedding_size)
    stacked = np.stack(unstacked)
    return stacked

def detokenize(tokenizer: PreTrainedTokenizerFast, id_seq: List[int]) -> List[str]:
    full_string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(id_seq, skip_special_tokens=True))
    target_names = full_string.split(TARGET_DELIMITER)
    return target_names

def detokenize_generated_targets(generate_output: GenerateBeamEncoderDecoderOutput,
                                 tokenizer: PreTrainedTokenizerFast) -> Tuple[List[str], List[int]]:
    output_ids = generate_output.sequences
    sample_inds = []
    all_texts = []
    for i, id_seq in enumerate(output_ids.detach().cpu().tolist()):
        sample_targets = detokenize(tokenizer, id_seq)
        all_texts.extend(sample_targets)
        sample_inds.extend(i for _ in sample_targets)
    return all_texts, sample_inds


class TargetPredictionWriter(BasePredictionWriter, TargetMixin):
    def __init__(self,
                 out_dir: os.PathLike,
                 embeddings_path: os.PathLike,
                 targets_path: os.PathLike,
                 map_generated: bool = False,
                 related_threshold: float = DEFAULT_RELATED_THRESHOLD,
                 ):
        BasePredictionWriter.__init__(self, write_interval='batch')
        TargetMixin.__init__(self, targets_path)
        self.out_dir = out_dir
        self.map_generated = map_generated

        if self.map_generated:
            self.related_threshold = related_threshold
            self.fast_text = FastText.load(str(embeddings_path))
        self.__target_embeddings = torch.tensor(make_target_embeddings(self.targets, self.fast_text), device='cpu')

        self.__device: Optional[str] = None

        self.__started_file = set()

        self.__gen_fieldnames = ["Sample", "Generated Target", "GT Target"]
        self.__map_fieldnames = ["Sample", "Generated Target", "Mapped Target", "GT Target"]

    def _map_targets(self, all_texts: List[str], sample_inds: List[int]) -> Tuple[torch.Tensor, List[str]]:
        fast_text = self.fast_text
        device = self.__device
        target_embeddings = self.__target_embeddings
        related_threshold = self.related_threshold

        sample_inds = torch.tensor(sample_inds, dtype=torch.long, device=device)
        output_embeddings = fast_text.wv[all_texts]
        output_embeddings = torch.tensor(output_embeddings).to(device)
        output_embeddings = output_embeddings / torch.linalg.norm(output_embeddings, keepdim=True, dim=1)

        # For a given sample, we want to pick the fixed target that had the highest similarity score,
        # across any predicted target for that sample

        # 1. Calculate the similarity scores
        # (n_predicted_targets, n_fixed_targets)
        all_scores = output_embeddings @ target_embeddings.transpose(1, 0)
        # 2. Calculate the highest score for a given fixed target across the predicted targets for a sample
        # Different samples can have different numbers of targets predicted, so we have to use a sparse operation here
        # (n_samples, n_fixed_targets)
        sample_scores, arg_sample_scores = segment_max_coo(all_scores, sample_inds)
        # 3. Take the max across the fixed targets
        # (n_samples,) and (n_samples,)
        class_scores, arg_class_scores = torch.max(sample_scores, dim=-1)
        # 4. Pick the UNRELATED target if the class score doesn't meet the similarity threshold
        target_preds = torch.where(class_scores >= related_threshold, arg_class_scores, 0)

        # Get the string-based free-form targets as well
        all_text_inds = torch.gather(arg_sample_scores, arg_class_scores.unsqueeze(1), dim=1).squeeze(1)
        freeform_preds = [all_texts[i] for i in all_text_inds]

        return target_preds, freeform_preds


    @staticmethod
    def __cons_writer(file_handle, fieldnames):
        return csv.DictWriter(file_handle, fieldnames=fieldnames, lineterminator='\n')

    @contextmanager
    def __get_writer(self, out_path, fieldnames, dataloader_idx):
        if dataloader_idx in self.__started_file:
            try:
                with open(out_path, 'a') as w:
                    yield self.__cons_writer(w, fieldnames)
            finally:
                pass
        else:
            self.__started_file.add(dataloader_idx)
            try:
                with open(out_path, 'w') as w:
                    writer = self.__cons_writer(w, fieldnames)
                    writer.writeheader()
                    yield writer
            finally:
                pass

    def __get_gen_writer(self, dataloader_idx):
        return self.__get_writer(
            os.path.join(self.out_dir, f"target_gens.{dataloader_idx}.txt"),
            self.__gen_fieldnames,
            dataloader_idx
        )

    def __get_map_writer(self, dataloader_idx):
        return self.__get_writer(
            os.path.join(self.out_dir, f"target_preds.{dataloader_idx}.txt"),
            self.__map_fieldnames,
            dataloader_idx
        )

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if self.target_level <= TargetLevel.NONE:
            return
        if self.__device is None:
            self.__device = pl_module.device
            self.__target_embeddings = self.__target_embeddings.to(self.__device)

        target_labels = batch['target'].flatten().detach().cpu().tolist()
        str_labels = [target_lookup[t] for t in target_labels]

        all_texts, sample_inds = detokenize_generated_targets(prediction, pl_module.tokenizer)

        row_dicts = [{"Sample": i, "Generated Target": text, "GT Target": str_labels[i]} for i,text in zip(all_texts, sample_inds) ]
        with self.__get_gen_writer(dataloader_idx) as writer:
            writer.writerows(row_dicts)


        target_preds = prediction.target_preds.flatten().detach().cpu().tolist()
        target_lookup = pl_module.targets
        str_preds = [target_lookup[t] for t in target_preds]
        row_dicts = [{"Mapped Target": pred, "GT Target": label} for pred, label in zip(str_preds, str_labels)]
        with self.__get_writer(dataloader_idx) as writer:
            writer.writerows(row_dicts)
