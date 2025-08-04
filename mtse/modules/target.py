from __future__ import annotations
import pathlib
import dataclasses
from typing import Optional
# 3rd Party
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
from transformers import BertweetTokenizer, RobertaModel
# 
from .base_module import BaseModule
from ..data import Encoder, Sample, collate_ids, keyed_scalar_stack, try_add_position_ids
from ..constants import DEFAULT_HF_MODEL

class TargetModule(BaseModule):

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor

    def __init__(self, targets_path: pathlib.Path):
        super().__init__()
        self.no_target = 0
        with open(targets_path, 'r') as r:
            targets = [t.strip() for t in r]
        self.targets = targets
    
    @property
    def n_targets(self):
        return len(self.targets)

    @property
    def max_length(self) -> Optional[int]:
        return None

    def training_step(self, batch, batch_idx):
        target_logits = self(**batch)
        target_loss = torch.nn.functional.cross_entropy(target_logits, batch['target'])
        self.log('loss/target', target_loss)
        loss = target_loss
        self.log('loss', loss)
        return loss

    def _infer_step(self, batch):
        target_logits = self(**batch)
        return TargetModule.Output(
            target_preds=torch.argmax(target_logits, dim=-1),
        )

    class Encoder(Encoder):
        def __init__(self, module: TargetModule):
            self.module = module
            self.tokenizer: PreTrainedTokenizerFast = module.tokenizer
        def encode(self, sample: Sample, inference=False):
            encoding = self.tokenizer(text=sample.context, return_tensors='pt',
                                      truncation=True, max_length=self.module.max_length)
            try_add_position_ids(encoding)
            # +1 to handle the nontarget-0
            target_code = 0 if sample.target is None else self.module.targets.index(sample.target) + 1
            encoding['target'] = torch.tensor(target_code)
            return encoding
        def collate(self, samples):
            rdict = collate_ids(self.module.tokenizer, samples, return_attention_mask=True)
            rdict['target'] = keyed_scalar_stack(samples, 'target')
            return rdict

class YingjieTargetModule(TargetModule):
    """
    Modelled from code in paper https://aclanthology.org/2023.acl-long.560/

    Had some very specific gradient-clipping and learning rate settings, so easier to
    implement in a separate class
    """

    PRETRAINED_MODEL = "vinai/bertweet-base"

    def __init__(self, **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.bert = RobertaModel.from_pretrained(YingjieTargetModule.PRETRAINED_MODEL)
        self.tokenizer = BertweetTokenizer.from_pretrained(YingjieTargetModule.PRETRAINED_MODEL)
        config = self.bert.config
        self.__max_length: Optional[int] = getattr(config, "max_position_embeddings", None)
        hidden_size = config.hidden_size

        self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.out = torch.nn.Linear(hidden_size, self.n_targets + 1)
        self.__encoder = self.Encoder(self)

    def configure_optimizers(self):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        return torch.optim.AdamW([
           {"params": self.linear.parameters(), "lr": 1e-3},
           {"params": self.out.parameters(), "lr": 1e-3},
           {"params": self.bert.pooler.parameters(), "lr": 1e-3},
           {"params": self.bert.encoder.parameters(), "lr": 2e-5}
        ])

    @property
    def max_length(self):
        return self.__max_length

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, **kwargs):
        bert_kwargs = {k:v for k,v in kwargs.items() if k != 'target' and k != 'stance'}
        bert_output = self.bert(**bert_kwargs)
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        # Don't use torch.nn.Sequential because I want invididual layer variables I can refer to
        # in configure_optimizers()
        target_logits = self.out(torch.nn.functional.relu(self.linear(cls_hidden_state)))
        return target_logits


class HFTargetModule(TargetModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_HF_MODEL,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        config = self.bert.config

        self.__max_length: Optional[int] = getattr(config, "max_position_embeddings", None)

        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        hidden_size = config.hidden_size
        self.target_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.n_targets + 1)
        )
        self.__encoder = self.Encoder(self)

    @property
    def max_length(self):
        return self.__max_length

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, **kwargs):
        bert_kwargs = {k:v for k,v in kwargs.items() if k != 'target' and k != 'stance'}
        bert_output = self.bert(**bert_kwargs)
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        target_logits = self.target_classifier(cls_hidden_state)
        return target_logits

__all__ = ["TargetModule", "YingjieTargetModule", "HFTargetModule"]