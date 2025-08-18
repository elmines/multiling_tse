from __future__ import annotations
import pathlib
import dataclasses
from typing import Optional
# 3rd Party
import numpy as np
import torch
from transformers import RobertaModel, PreTrainedTokenizerFast, BertweetTokenizer, BartModel, BartTokenizerFast
from gensim.models import FastText
# 
from .base_module import BaseModule
from ..data import Encoder, StanceType, STANCE_TYPE_MAP, Sample, collate_ids, keyed_scalar_stack, try_add_position_ids
from ..constants import DEFAULT_HF_MODEL, UNRELATED_TARGET

class OneShotModule(BaseModule):

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor
        stance_preds: torch.Tensor

    def __init__(self,
                 targets_path: pathlib.Path,
                 stance_type: StanceType):
        super().__init__()
        self.stance_type = STANCE_TYPE_MAP[stance_type]
        with open(targets_path, 'r') as r:
            targets = [t.strip() for t in r]
        self.targets = [UNRELATED_TARGET] + targets
    
    @property
    def n_targets(self):
        return len(self.targets)

class LiOneShotModule(OneShotModule):
    """
    One-shot module designed to be as close as possible to Li et al.'s
    individual target and stance models
    """
    PRETRAINED_MODEL = "vinai/bertweet-base"

    NON_BERT_KEYS = {'target', 'stance'}

    def __init__(self, **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.bert = RobertaModel.from_pretrained(LiOneShotModule.PRETRAINED_MODEL)
        self.tokenizer = BertweetTokenizer.from_pretrained(LiOneShotModule.PRETRAINED_MODEL, normalization=True)
        config = self.bert.config
        hidden_size = config.hidden_size

        self.stance_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(self.stance_type))
        )
        self.target_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(self.targets))
        )

        self.__encoder = self.Encoder(self)

    @property
    def encoder(self):
        return self.__encoder

    def configure_optimizers(self):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        return torch.optim.AdamW([
           {"params": self.stance_classifier.parameters(), "lr": 1e-3},
           {"params": self.target_classifier.parameters(), "lr": 1e-3},
           {"params": self.bert.encoder.parameters(), "lr": 2e-5},
        ])

    def configure_gradient_clipping(self, optimizer, gradient_clip_val = None, gradient_clip_algorithm = None):
        assert gradient_clip_val is None and gradient_clip_algorithm is None
        return super().configure_gradient_clipping(optimizer, 1.0, 'norm')


    def forward(self, **kwargs):
        bert_kwargs = {k:v for k,v in kwargs.items() if k not in LiOneShotModule.NON_BERT_KEYS}
        bert_output = self.bert(**bert_kwargs)
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        target_logits = self.target_classifier(cls_hidden_state)
        stance_logits = self.stance_classifier(cls_hidden_state)
        return target_logits, stance_logits

    def training_step(self, batch, batch_idx):
        target_logits, stance_logits = self(**batch)
        target_loss = torch.nn.functional.cross_entropy(target_logits, batch['target'])
        stance_loss = torch.nn.functional.cross_entropy(stance_logits, batch['stance'])
        loss = target_loss + stance_loss
        self.log('loss/target', target_loss)
        self.log('loss/stance', stance_loss)
        self.log('loss', loss)
        return loss

    def _infer_step(self, batch):
        target_logits, stance_logits = self(**batch)
        return OneShotModule.Output(
            target_preds=torch.argmax(target_logits, dim=-1),
            stance_preds=torch.argmax(stance_logits, dim=-1)
        )

    class Encoder(Encoder):
        def __init__(self, module: LiOneShotModule):
            self.module = module
            self.tokenizer: PreTrainedTokenizerFast = module.tokenizer
        def encode(self, sample: Sample, inference=False):
            encoding = self.tokenizer(text=sample.context, return_tensors='pt',
                                    truncation=True, max_length=128)
            encoding['target'] = torch.tensor(self.module.targets.index(sample.target_label))
            encoding['stance'] = torch.tensor(sample.stance)
            return encoding
        def collate(self, samples):
            rdict = collate_ids(self.module.tokenizer, samples, return_attention_mask=True)
            rdict['target'] = keyed_scalar_stack(samples, 'target')
            rdict['stance'] = keyed_scalar_stack(samples, 'stance')
            return rdict

class TGOneShotModule(OneShotModule):

    PRETRAINED_MODEL = "facebook/bart-base"

    def __init__(self,
                 embeddings_path: pathlib.Path,
                 related_threshold: float = 0.2,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.related_threshold = related_threshold

        self.bart = BartModel.from_pretrained(TGOneShotModule.PRETRAINED_MODEL)
        self.tokenizer = BartTokenizerFast.from_pretrained(TGOneShotModule.PRETRAINED_MODEL, normalization=True)

        config = self.bart.config
        hidden_size = config.hidden_size
        self.stance_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(self.stance_type))
        )

        self.fast_text = FastText.load(str(embeddings_path))

        unstacked = []
        for target in self.targets:
            if target == UNRELATED_TARGET:
                # Will never match with the unrelated target, because no vector X can have
                # a nonzero cosine similarity with the zero vector
                unstacked.append(np.zeros([self.fast_text.vector_size], dtype=np.float32))
            else:
                unstacked.append(self.fast_text.wv[target.lower()])
        stacked = np.stack(unstacked)

        self.target_embeddings: torch.Tensor # Only have this so the IDE knows this variable exists
        self.register_buffer("target_embeddings", torch.tensor(stacked), persistent=False)


__all__ = ["OneShotModule", "LiOneShotModule", "TGOneShotModule"]