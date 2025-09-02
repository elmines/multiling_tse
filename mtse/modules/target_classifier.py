from __future__ import annotations
import pathlib
import dataclasses
from typing import Optional
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BertweetTokenizer, RobertaModel
# 
from .base_module import BaseModule
from .mixins import TargetMixin
from ..data import Encoder, PredictTask, Sample, collate_ids, keyed_scalar_stack, try_add_position_ids

class _TargetClassifierModule(BaseModule, TargetMixin):

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor

    def __init__(self, targets_path: pathlib.Path):
        BaseModule.__init__(self)
        TargetMixin.__init__(self, targets_path)

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
        return _TargetClassifierModule.Output(
            target_preds=torch.argmax(target_logits, dim=-1),
        )

    class Encoder(Encoder):
        def __init__(self, module: _TargetClassifierModule):
            super().__init__()
            self.module = module
            self.tokenizer: PreTrainedTokenizerFast = module.tokenizer
        def _encode(self, sample: Sample, inference=False, predict_task: Optional[PredictTask] = None):
            assert predict_task is None or predict_task == PredictTask.TARGET
            encoding = self.tokenizer(text=sample.context, return_tensors='pt',
                                      truncation=True, max_length=self.module.max_length)
            try_add_position_ids(encoding)
            target_code = self.module.targets.index(sample.target)
            encoding['target'] = torch.tensor(target_code)
            return encoding
        def collate(self, samples):
            rdict = collate_ids(self.module.tokenizer, samples, return_attention_mask=True)
            rdict['target'] = keyed_scalar_stack(samples, 'target')
            return rdict

class LiTargetClassifierModule(_TargetClassifierModule):
    """
    Modelled from code in paper https://aclanthology.org/2023.acl-long.560/

    Had some very specific gradient-clipping and learning rate settings, so easier to
    implement in a separate class
    """

    PRETRAINED_MODEL = "vinai/bertweet-base"

    def __init__(self, **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.bert = RobertaModel.from_pretrained(LiTargetClassifierModule.PRETRAINED_MODEL)
        self.tokenizer = BertweetTokenizer.from_pretrained(LiTargetClassifierModule.PRETRAINED_MODEL, normalization=True)
        config = self.bert.config
        self.__max_length: Optional[int] = getattr(config, "max_position_embeddings", None)
        hidden_size = config.hidden_size

        self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.out = torch.nn.Linear(hidden_size, self.n_targets)
        self.__encoder = self.Encoder(self)

    class Encoder(_TargetClassifierModule.Encoder):
        def _encode(self, sample: Sample, inference=False, predict_task: Optional[PredictTask] = None):
            assert predict_task is None or predict_task == PredictTask.TARGET
            assert sample.target_label is not None
            # This looks clunky, but trying to imitate Li's original code
            encoding = self.tokenizer.encode_plus(text=sample.context, 
                                      padding="max_length", max_length=128, add_special_tokens=True,
                                      return_attention_mask=True)
            encoding = {k:torch.unsqueeze(torch.tensor(v, dtype=torch.long), 0) for k,v in encoding.items()}
            
            keys = list(encoding)
            for k in keys:
                if encoding[k].shape[-1] > 128:
                    encoding[k] = encoding[k][..., :128]
            target_code = self.module.targets.index(sample.target_label)
            encoding['target'] = torch.tensor(target_code)
            return encoding

    def configure_optimizers(self):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        return torch.optim.AdamW([
           {"params": self.linear.parameters(), "lr": 1e-3},
           {"params": self.out.parameters(), "lr": 1e-3},
           # Don't think the pooler is actually used,
           # but just for consistency with Li et al.'s code leaving this
           {"params": self.bert.pooler.parameters(), "lr": 1e-3},
           {"params": self.bert.encoder.parameters(), "lr": 2e-5}
        ])

    @property
    def max_length(self):
        return self.__max_length

    @property
    def encoder(self):
        return self.__encoder

    def training_step(self, batch, batch_idx):
        target_logits = self(**batch)
        # Li specifically chose a sum reduction, not the 'mean' default
        target_loss = torch.nn.functional.cross_entropy(target_logits, batch['target'], reduction='sum')
        self.log('loss/target', target_loss)
        loss = target_loss
        self.log('loss', loss)
        return loss

    def forward(self, **kwargs):
        bert_kwargs = {k:v for k,v in kwargs.items() if k != 'target' and k != 'stance'}
        bert_output = self.bert(**bert_kwargs)
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        # Don't use torch.nn.Sequential because I want invididual layer variables I can refer to
        # in configure_optimizers()
        target_logits = self.out(torch.nn.functional.relu(self.linear(cls_hidden_state)))
        return target_logits


__all__ = ["_TargetClassifierModule", "LiTargetClassifierModule"]