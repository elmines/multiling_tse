from __future__ import annotations
import dataclasses
import pathlib
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast
# Local
from .base_module import BaseModule
from ..data import StanceType, STANCE_TYPE_MAP, Encoder, Sample, collate_ids, keyed_scalar_stack

class StanceOnlyModule(BaseModule):
    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor
        stance_preds: torch.Tensor

    def __init__(self,
                 targets_path: pathlib.Path,
                 stance_type: StanceType,
                 pretrained_model: str = "google-bert/bert-base-chinese"):
        super().__init__()
        self.stance_type = STANCE_TYPE_MAP[stance_type]
        self.no_target = 0
        with open(targets_path, 'r') as r:
            targets = [t.strip() for t in r]
        self.targets = targets

        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        config = self.bert.config
        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        hidden_size = config.hidden_size
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
        stance_logits = self.stance_classifier(cls_hidden_state)
        return stance_logits

    def training_step(self, batch, batch_idx):
        stance_logits = self(**batch)
        loss = torch.nn.functional.cross_entropy(stance_logits, batch['stance'])
        self.log("loss/stance", loss)
        self.log("loss", loss)
        return loss

    def _infer_step(self, batch):
        stance_logits = self(**batch)
        return self.Output(
            target_preds=batch['target'], # This module gets to cheat and use the gold targets
            stance_preds=torch.argmax(stance_logits, dim=1)
        )

    class Encoder(Encoder):
        def __init__(self, module: StanceOnlyModule):
            self.module = module
            self.tokenizer = module.tokenizer
        def encode(self, sample: Sample, inference=False):
            target_text = sample.target if sample.target else 'unknown'
            encoding = self.tokenizer(text=sample.context, text_pair=target_text, return_tensors='pt')
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
