from __future__ import annotations
import dataclasses
import pathlib
from typing import Optional
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast, BertweetTokenizer, RobertaModel
# Local
from .base_module import BaseModule
from ..data import StanceType, STANCE_TYPE_MAP, Encoder, PredictTask, Sample, collate_ids, keyed_scalar_stack
from ..constants import UNRELATED_TARGET

class TwoShotModule(BaseModule):
    @dataclasses.dataclass
    class Output:
        target_preds: Optional[torch.Tensor] = None
        stance_preds: Optional[torch.Tensor] = None

    def __init__(self,
                 targets_path: pathlib.Path,
                 stance_type: StanceType,
                 use_target_gt: bool = False):
        super().__init__()
        self.stance_type = STANCE_TYPE_MAP[stance_type]
        self.no_target = 0
        with open(targets_path, 'r') as r:
            targets = [t.strip() for t in r]
        self.targets = [UNRELATED_TARGET] + targets
        self.use_target_gt = use_target_gt

class LiTwoShotModule(TwoShotModule):

    PRETRAINED_MODEL = "vinai/bertweet-base"

    NON_BERT_KEYS = {'target', 'target_in', 'stance', 'task'}

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor
        stance_preds: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None
        
    def __init__(self, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.bert = RobertaModel.from_pretrained(LiTwoShotModule.PRETRAINED_MODEL)
        self.tokenizer = BertweetTokenizer.from_pretrained(LiTwoShotModule.PRETRAINED_MODEL, normalization=True)
        config = self.bert.config
        self.max_length: int = 128
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

        self.__last_task: int = -1

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
        if self.__last_task == PredictTask.STANCE:
            return super().configure_gradient_clipping(optimizer, 1.0, 'norm')
        elif self.__last_task == PredictTask.TARGET:
            return super().configure_gradient_clipping(optimizer, 0.75, 'norm')
        else:
            raise ValueError(f"Invalid task ID {self.__last_task}")

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, **kwargs):
        bert_kwargs = {k:v for k,v in kwargs.items() if k not in LiTwoShotModule.NON_BERT_KEYS}
        bert_output = self.bert(**bert_kwargs)
        cls_hidden_state = bert_output.last_hidden_state[:, 0]

        task = kwargs.get('task', PredictTask.STANCE)
        self.__last_task = task
        if task == PredictTask.STANCE:
            stance_logits = self.stance_classifier(cls_hidden_state)
            # Li uses a mean reduction in the stance training,
            # not a sum reduction like in the target classifier training
            loss = torch.nn.functional.cross_entropy(stance_logits, kwargs['stance'])

            target_preds = kwargs['target' if self.use_target_gt else 'target_in']
            return LiTwoShotModule.Output(target_preds=target_preds,
                                          stance_preds=torch.argmax(stance_logits, dim=1),
                                          loss=loss)
        else:
            target_logits = self.target_classifier(cls_hidden_state)
            loss = torch.nn.functional.cross_entropy(target_logits, kwargs['target'])
            return LiTwoShotModule.Output(target_preds=torch.argmax(target_logits, dim=1),
                                          loss=loss)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        self.log("loss", loss)
        return loss

    def _infer_step(self, batch):
        return self(**batch)

    class Encoder(Encoder):
        def __init__(self, module: LiTwoShotModule):
            self.module = module
            self.tokenizer: PreTrainedTokenizerFast = module.tokenizer

        def encode(self, sample: Sample, inference=False, predict_task: Optional[PredictTask] = None):
            if predict_task is None:
                predict_task = PredictTask.STANCE

            if predict_task == PredictTask.STANCE:
                assert sample.target_input is not None
                encoding = self.tokenizer(
                                        text=sample.target_input,
                                        text_pair=sample.context,
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=self.module.max_length,
                                        padding='max_length',
                                        return_attention_mask=True)
                encoding['target_in'] = torch.tensor(self.module.targets.index(sample.target_input))
            elif predict_task == PredictTask.TARGET:
                encoding = self.tokenizer(text=sample.context, return_tensors='pt',
                                          truncation=True, max_length=self.module.max_length, padding='max_length',
                                          return_attention_mask=True)
            else:
                raise ValueError(f"Invalid task ID {predict_task}")
            assert sample.target_label is not None
            encoding['target'] = torch.tensor(self.module.targets.index(sample.target_label))
            encoding['stance'] = torch.tensor(sample.stance)
            encoding['task'] = torch.tensor(predict_task, dtype=torch.long)
            return encoding
        def collate(self, samples):
            tasks = [s.get('task') for s in samples]
            if not all(t is not None and t == tasks[0] for t in tasks):
                raise ValueError("Need matching task IDs for all samples in batch")
            rdict = collate_ids(self.module.tokenizer, samples, return_attention_mask=True)
            for scalar_key in ['target', 'stance']:
                rdict[scalar_key] = keyed_scalar_stack(samples, scalar_key)
            for scalar_key in filter(lambda k: k in samples[0], ['target_in']):
                rdict[scalar_key] = keyed_scalar_stack(samples, scalar_key)
            rdict['task'] = tasks[0]
            return rdict
