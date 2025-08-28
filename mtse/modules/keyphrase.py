from __future__ import annotations
import dataclasses
# 3rd Party
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizerFast
# Local
from .base_module import BaseModule
from ..data import Encoder, Sample, collate_ids

class BartKeyphraseModule(BaseModule):

    @dataclasses.dataclass
    class TrainOutput:
        lm_loss: torch.Tensor
        stance_loss: torch.Tensor

    PRETRAINED_MODEL = "facebook/bart-base"

    def __init__(self,
                 backbone_lr: float = 1e-5,
                 head_lr: float = 4e-5,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.bart = BartForConditionalGeneration.from_pretrained(BartKeyphraseModule.PRETRAINED_MODEL)
        self.tokenizer: PreTrainedTokenizerFast = BartTokenizerFast.from_pretrained(BartKeyphraseModule.PRETRAINED_MODEL, normalization=True)
        self.__encoder = self.Encoder(self)

    def configure_optimizers(self):
        lm_head_params = set(self.bart.lm_head.parameters())
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
    def validation_step(self, batch, batch_idx):
        res = self(**batch)
        loss = res.loss
        self.log('val/loss', loss)
        return loss

    class Encoder(Encoder):
        def __init__(self, module: BartKeyphraseModule):
            self.module = module
            self.tokenizer = module.tokenizer
            self.max_length = self.module.bart.config.max_position_embeddings
        def _encode(self, sample: Sample, inference=False, predict_task = None):
            return self.tokenizer(text=sample.context.lower(),
                                      text_target=sample.target_label.lower(),
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=self.max_length)
        def collate(self, samples):
            return collate_ids(self.tokenizer, samples, return_attention_mask=True)

