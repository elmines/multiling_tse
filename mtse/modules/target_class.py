from __future__ import annotations
import pathlib
import dataclasses
# 3rd Party
import torch
from transformers import AutoModel, AutoTokenizer
# 
from .base_module import BaseModule
from ..data import Encoder, StanceType, STANCE_TYPE_MAP, Sample, collate_ids, keyed_scalar_stack

class TargetClassModule(BaseModule):

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor
        stance_preds: torch.Tensor

    def __init__(self,
                 targets_path: pathlib.Path,
                 stance_type: StanceType):
        super().__init__()
        self.stance_type = STANCE_TYPE_MAP[stance_type]
        self.no_target = 0
        with open(targets_path, 'r') as r:
            targets = [t.strip() for t in r]
        self.targets = targets
    
    @property
    def n_targets(self):
        return len(self.targets)

class BertTCModule(TargetClassModule):
    def __init__(self,
                 pretrained_model: str = "google-bert/bert-base-chinese",
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        config = self.bert.config
        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        hidden_size = config.hidden_size
        self.target_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.n_targets + 1)
        )
        self.stance_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(self.stance_type))
        )
        self.__encoder = self.Encoder(self)

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, **kwargs):
        bert_kwargs = {k:v for k,v in kwargs.items() if k != 'target' and k != 'stance'}
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
        return TargetClassModule.Output(
            target_preds=torch.argmax(target_logits, dim=-1),
            stance_preds=torch.argmax(stance_logits, dim=-1)
        )

    class Encoder(Encoder):
        def __init__(self, module: BertTCModule):
            self.module = module
            self.tokenizer = module.tokenizer
        def encode(self, sample: Sample, inference=False):
            encoding = self.tokenizer(text=sample.context, return_tensors='pt')
            # +1 to handle the nontarget-0
            target_code = 0 if sample.target is None else self.module.targets.index(sample.target) + 1
            encoding['target'] = torch.tensor(target_code)
            encoding['stance'] = torch.tensor(sample.stance)
            return encoding
        def collate(self, samples):
            rdict = collate_ids(self.module.tokenizer, samples, return_attention_mask=True)
            rdict['target'] = keyed_scalar_stack(samples, 'target')
            rdict['stance'] = keyed_scalar_stack(samples, 'stance')
            return rdict

__all__ = ["TargetClassModule", "BertTCModule"]