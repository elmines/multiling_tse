from __future__ import annotations
import dataclasses
import typing
from typing import List
import pathlib
import functools
# 3rd Party
from gensim.models import FastText
import numpy as np
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizerFast, MBart50Tokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from torch_scatter import segment_max_coo
# Local
from .base_module import BaseModule
from .mixins import TargetMixin
from ..data import Encoder, Sample, collate_ids, keyed_scalar_stack, SampleType
from ..constants import UNRELATED_TARGET, TARGET_DELIMITER, DEFAULT_RELATED_THRESHOLD

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

def pick_targets(generate_output: GenerateBeamEncoderDecoderOutput,
                 tokenizer: PreTrainedTokenizerFast,
                 fast_text: FastText,
                 target_embeddings: torch.Tensor,
                 related_threshold: float):
    """
    :param target_embeddings: (n_targets, embedding_size) tensor
    """
    output_ids = generate_output.sequences
    sample_inds = []
    all_texts = []
    for i, id_seq in enumerate(output_ids.detach().cpu().tolist()):
        sample_targets = detokenize(tokenizer, id_seq)
        all_texts.extend(sample_targets)
        sample_inds.extend(i for _ in sample_targets)
    sample_inds = torch.tensor(sample_inds, dtype=torch.long, device=output_ids.device)
    output_embeddings = fast_text.wv[all_texts]
    output_embeddings = torch.tensor(output_embeddings).to(output_ids.device)
    output_embeddings = output_embeddings / torch.linalg.norm(output_embeddings, keepdim=True, dim=1)

    # For a given sample, we want to pick the fixed target that had the highest similarity score,
    # across any predicted target for that sample

    # 1. Calculate the similarity scores
    # (n_predicted_targets, n_fixed_targets)
    all_scores = output_embeddings @ target_embeddings.transpose(1, 0)
    # 2. Calculate the highest score for a given fixed target across the predicted targets for a sample
    # Different samples can have different numbers of targets predicted, so we have to use a sparse operation here
    # (n_samples, n_fixed_targets)
    sample_scores, _ = segment_max_coo(all_scores, sample_inds)
    # 3. Take the max across the fixed targets
    # (n_samples,) and (n_samples,)
    class_scores, class_inds = torch.max(sample_scores, dim=-1)
    # 4. Pick the UNRELATED target if the class score doesn't meet the similarity threshold
    target_preds = torch.where(class_scores >= related_threshold, class_inds, 0)
    return target_preds


class LiTargetGenerator(BaseModule, TargetMixin):

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor

    PRETRAINED_MODEL = "facebook/bart-base"

    def __init__(self,
                 embeddings_path: pathlib.Path,
                 targets_path: pathlib.Path,
                 backbone_lr: float = 1e-5,
                 head_lr: float = 4e-5,
                 max_length: int = 75,
                 predict_targets: bool = False,
                 related_threshold: float = DEFAULT_RELATED_THRESHOLD,
                 multilingual: bool = False,
                 **parent_kwargs):
        BaseModule.__init__(self, **parent_kwargs)
        TargetMixin.__init__(self, targets_path)
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.max_length = max_length
        self.predict_targets = predict_targets
        self.related_threshold = related_threshold
        self.multilingual = multilingual
        if self.multilingual:
            self.bart = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
            self.tokenizer: PreTrainedTokenizerFast = T5Tokenizer.from_pretrained("google/mt5-base", normalization=True)
        else:
            self.bart = BartForConditionalGeneration.from_pretrained(LiTargetGenerator.PRETRAINED_MODEL)
            self.tokenizer: PreTrainedTokenizerFast = BartTokenizerFast.from_pretrained(LiTargetGenerator.PRETRAINED_MODEL, normalization=True)
        self.__encoder = self.Encoder(self)

        self.fast_text = FastText.load(str(embeddings_path))
        self.target_embeddings: torch.Tensor # Only have this so the IDE knows this variable exists
        self.register_buffer("target_embeddings", torch.tensor(make_target_embeddings(self.targets, self.fast_text)), persistent=False)

    def configure_optimizers(self):
        lm_head_params = set(self.bart.lm_head.parameters())
        if self.multilingual:
            excluded_params = set(self.bart.shared.parameters()) | lm_head_params
            backbone_params = [p for p in self.bart.parameters() if p not in excluded_params]
        else:
            # One overlapping parameter forces me to do this
            backbone_params = [p for p in self.bart.model.parameters() if p not in lm_head_params]
        lm_head_params = list(lm_head_params)
        return torch.optim.AdamW([
            {"params": lm_head_params, "lr": self.backbone_lr},
            {"params": backbone_params, "lr": self.head_lr},
        ])

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, **bart_kwargs):
        return self.bart(**bart_kwargs)
    def training_step(self, batch, batch_idx):
        res = self(**batch)
        loss = res.loss
        self.log('train/loss', loss)
        return loss

    def _predict_targets(self, batch):
        generate_output = self.bart.generate(batch['input_ids'],
                                         return_dict_in_generate=True,
                                         output_hidden_states=True,
                                         max_length=self.max_length,
                                         num_beams=3)
        target_preds = pick_targets(generate_output,
                                    self.tokenizer,
                                    self.fast_text,
                                    self.target_embeddings,
                                    self.related_threshold)
        return LiTargetGenerator.Output(target_preds)

    def validation_step(self, batch, batch_idx):
        res = self(**batch)
        self.log('val/loss', res.loss)
        if self.predict_targets:
            res = None # Want that entire forward pass gone to save memory for the target predictions
            # FIXME: Would be much more efficient to use the encoder hidden_states from res in _predict_targets
            return self._predict_targets(batch)

    def _infer_step(self, batch):
        if self.predict_targets:
            return self._predict_targets(batch)

    class Encoder(Encoder):
        def __init__(self, module: LiTargetGenerator):
            super().__init__()
            self.module = module
            self.tokenizer = module.tokenizer
            config = self.module.bart.config
            self.max_length = getattr(config, "max_position_embeddings", 1024)

        def _encode(self, sample: Sample, inference=False, predict_task = None):
            encoding = self.tokenizer(text=sample.context.lower(),
                                      text_target=sample.target_label.lower(),
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=self.max_length)
            if sample.sample_type == SampleType.SD:
                encoding['target'] = torch.tensor(
                    [self.module.targets.index(sample.target_label)],
                    dtype=torch.long)
            return encoding
        def collate(self, samples):
            encoding = collate_ids(self.tokenizer, samples, return_attention_mask=True)
            if "target" in samples[0]:
                encoding['target'] = keyed_scalar_stack(samples, 'target')
            return encoding

__all__ = ["make_target_embeddings", "detokenize", "pick_targets", "LiTargetGenerator"]