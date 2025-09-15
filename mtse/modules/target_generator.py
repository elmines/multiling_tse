from __future__ import annotations
# 3rd Party
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizerFast, MBart50Tokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
# Local
from .base_module import BaseModule
from ..data import Encoder, Sample, collate_ids, keyed_scalar_stack, SampleType

class LiTargetGenerator(BaseModule):

    PRETRAINED_MODEL = "facebook/bart-base"

    def __init__(self,
                 backbone_lr: float = 1e-5,
                 head_lr: float = 4e-5,
                 max_length: int = 75,
                 # TODO: Remove this silly type union needed for backwards compatibility
                 predict_targets: bool = False,
                 multilingual: bool = False,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.max_length = max_length
        self.predict_targets = predict_targets
        self.multilingual = multilingual
        if self.multilingual:
            self.bart = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
            self.tokenizer: PreTrainedTokenizerFast = T5Tokenizer.from_pretrained("google/mt5-base", normalization=True)
        else:
            self.bart = BartForConditionalGeneration.from_pretrained(LiTargetGenerator.PRETRAINED_MODEL)
            self.tokenizer: PreTrainedTokenizerFast = BartTokenizerFast.from_pretrained(LiTargetGenerator.PRETRAINED_MODEL, normalization=True)
        self.__encoder = self.Encoder(self)

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
        return self.bart.generate(batch['input_ids'],
                                         return_dict_in_generate=True,
                                         output_hidden_states=True,
                                         max_length=self.max_length,
                                         num_beams=3)

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
            return self.tokenizer(text=sample.context.lower(),
                                      text_target=sample.target_label.lower(),
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=self.max_length)
        def collate(self, samples):
            return collate_ids(self.tokenizer, samples, return_attention_mask=True)

__all__ = ["LiTargetGenerator"]