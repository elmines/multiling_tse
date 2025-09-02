from __future__ import annotations
import pathlib
import dataclasses
# 3rd Party
import torch
from transformers import RobertaModel, PreTrainedTokenizerFast, BertweetTokenizer, BartForConditionalGeneration, BartTokenizerFast
# 
from .mixins import TargetMixin
from .base_module import BaseModule
from ..data import Encoder, StanceType, STANCE_TYPE_MAP, Sample, collate_ids, keyed_scalar_stack, SampleType
from ..constants import UNRELATED_TARGET, TARGET_DELIMITER

class LiBasedOneShotTClsModule(BaseModule, TargetMixin):
    """
    One-shot module doing target-clasification,
    designed to be as close as possible to Li et al.'s individual target and stance models.

    """

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor
        stance_preds: torch.Tensor


    PRETRAINED_MODEL = "vinai/bertweet-base"

    NON_BERT_KEYS = {'target', 'stance'}

    def __init__(self,
                 targets_path: pathlib.Path,
                 stance_type: StanceType,
                 use_target_gt: bool = False,
                 **parent_kwargs):
        BaseModule.__init__(self, **parent_kwargs)
        TargetMixin.__init__(self, targets_path)
        self.stance_type = STANCE_TYPE_MAP[stance_type]
        self.use_target_gt = use_target_gt
        self.bert = RobertaModel.from_pretrained(LiBasedOneShotTClsModule.PRETRAINED_MODEL)
        self.tokenizer = BertweetTokenizer.from_pretrained(LiBasedOneShotTClsModule.PRETRAINED_MODEL, normalization=True)
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
        bert_kwargs = {k:v for k,v in kwargs.items() if k not in LiBasedOneShotTClsModule.NON_BERT_KEYS}
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
        target_preds = batch['target'] if self.use_target_gt else torch.argmax(target_logits, dim=-1)
        return LiBasedOneShotTClsModule.Output(
            target_preds=target_preds,
            stance_preds=torch.argmax(stance_logits, dim=-1)
        )

    class Encoder(Encoder):
        def __init__(self, module: LiBasedOneShotTClsModule):
            super().__init__()
            self.module = module
            self.tokenizer: PreTrainedTokenizerFast = module.tokenizer
        def _encode(self, sample: Sample, inference=False):
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

