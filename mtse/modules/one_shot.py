from __future__ import annotations
import pathlib
import dataclasses
from typing import Optional
# 3rd Party
import numpy as np
import torch
from transformers import RobertaModel, PreTrainedTokenizerFast, BertweetTokenizer, BartForConditionalGeneration, BartTokenizerFast
from gensim.models import FastText
# 
from .base_module import BaseModule
from ..data import Encoder, StanceType, STANCE_TYPE_MAP, Sample, collate_ids, keyed_scalar_stack, SampleType
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

class ClassfierOneShotModule(OneShotModule):
    """
    One-shot module designed to be as close as possible to Li et al.'s
    individual target and stance models
    """
    PRETRAINED_MODEL = "vinai/bertweet-base"

    NON_BERT_KEYS = {'target', 'stance'}

    def __init__(self, **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.bert = RobertaModel.from_pretrained(ClassfierOneShotModule.PRETRAINED_MODEL)
        self.tokenizer = BertweetTokenizer.from_pretrained(ClassfierOneShotModule.PRETRAINED_MODEL, normalization=True)
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
        bert_kwargs = {k:v for k,v in kwargs.items() if k not in ClassfierOneShotModule.NON_BERT_KEYS}
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
        def __init__(self, module: ClassfierOneShotModule):
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

    @dataclasses.dataclass
    class TrainOutput:
        lm_loss: torch.Tensor
        stance_loss: torch.Tensor

    @dataclasses.dataclass
    class InferOutput:
        target_preds: torch.Tensor
        stance_preds: torch.Tensor

    PRETRAINED_MODEL = "facebook/bart-base"

    EXCLUDE_KWARGS = {'target', 'stance', 'stype', 'lm_weight'}

    def __init__(self,
                 embeddings_path: pathlib.Path,
                 related_threshold: float = 0.2,
                 backbone_lr: float = 1e-5,
                 head_lr: float = 4e-5,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.related_threshold = related_threshold
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr

        self.bart = BartForConditionalGeneration.from_pretrained(TGOneShotModule.PRETRAINED_MODEL)
        self.tokenizer: PreTrainedTokenizerFast = BartTokenizerFast.from_pretrained(TGOneShotModule.PRETRAINED_MODEL, normalization=True)

        config = self.bart.config
        hidden_size = config.hidden_size

        self.cross_att = torch.nn.MultiheadAttention(embed_dim=hidden_size,
                                                     num_heads=1,
                                                     batch_first=True)

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
                vec = self.fast_text.wv[target.lower()].copy()
                # Normalize magnitude to unity--makes computing explicit cosine similarity unnecessary
                # Just find the target vector that has the highest dot product with the predicted target
                vec = vec / np.linalg.norm(vec) 
                unstacked.append(vec)
        # (embedding_size, n_targets)
        stacked = np.stack(unstacked).transpose()

        self.target_embeddings: torch.Tensor # Only have this so the IDE knows this variable exists
        self.register_buffer("target_embeddings", torch.tensor(stacked), persistent=False)

        self.__encoder = self.Encoder(self)

        self.count_stance = 0
        self.count_kw = 0

    def configure_optimizers(self):
        lm_head_params = set(self.bart.lm_head.parameters())
        # One overlapping parameter forces me to do this
        backbone_params = [p for p in self.bart.model.parameters() if p not in lm_head_params]
        lm_head_params = list(lm_head_params)
        return torch.optim.AdamW([
            {"params": lm_head_params, "lr": self.backbone_lr},
            {"params": backbone_params, "lr": self.head_lr},
            {"params": self.stance_classifier.parameters(), "lr": self.head_lr},
            {"params": self.cross_att.parameters(), "lr": self.head_lr},
        ])

    @property
    def encoder(self):
        return self.__encoder

    def __detokenize(self, id_seq):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(id_seq, skip_special_tokens=True))

    def training_step(self, batch, batch_idx):
        bart_kwargs = {k:v for k,v in batch.items() if k not in TGOneShotModule.EXCLUDE_KWARGS}
        bart_output = self.bart(**bart_kwargs, output_hidden_states=True)

        lm_loss = bart_output.loss
        if batch['stype'] == SampleType.SD:
            encoder_hidden_states = bart_output.encoder_hidden_states[-1]
            decoder_hidden_states = bart_output.decoder_hidden_states[-1]
            target_features = self._pool_decoder_states(decoder_hidden_states, batch['labels'] != self.tokenizer.pad_token_id)
            stance_features = self._forward_att(encoder_hidden_states, target_features, batch['attention_mask'])
            stance_logits = self.stance_classifier(stance_features)
            stance_loss = torch.nn.functional.cross_entropy(stance_logits, batch['stance'])
            self.log('train/loss/stance', stance_loss)
            return stance_loss
        else:
            self.log('train/loss/lm', lm_loss)
            return lm_loss

    def _forward_att(self, encoder_hidden_states, target_features, attention_mask):
        att_out, _ = self.cross_att(
            query=torch.unsqueeze(target_features, dim=1),
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=torch.logical_not(attention_mask))
        feature_vec = torch.squeeze(att_out, dim=1)
        return feature_vec

    @staticmethod
    def _pool_decoder_states(decoder_hidden_states, not_padding):
        decoder_weights = not_padding.to(decoder_hidden_states.dtype)
        decoder_weights = torch.unsqueeze(decoder_weights, -1)
        num = torch.sum(decoder_weights * decoder_hidden_states, dim=-2)
        denom = torch.sum(decoder_weights, dim=-2)
        target_features = num / denom
        return target_features

    def _infer_step(self, batch):
        generate_output = self.bart.generate(batch['input_ids'],
                                             return_dict_in_generate=True,
                                             output_hidden_states=True,
                                             max_length=5,
                                             num_beams=3)

        # (beam_width * batch_size, seq_length, hidden_size)
        allbeam_states = torch.concatenate(
            [tok_states[-1] for tok_states in generate_output.decoder_hidden_states],
        dim=-2)

        # Have to duplicate that last dimension for .gather to work
        # (batch_size, seq_length, hidden_size)

        beam_indices = generate_output.beam_indices
        # .generate returns a -1 beam index for padding tokens
        # torch.gather does not accept negative indices
        # We arbitrarily replace them with 0, because ultimately those output token positions
        # will get ignored by _pool_decoder_states
        beam_indices = torch.where(beam_indices < 0, 0, beam_indices)
        beam_indices = torch.unsqueeze(beam_indices, -1).expand(-1, -1, allbeam_states.shape[-1])
        beam_indices = beam_indices.to(torch.int64)

        decoder_hidden_states = torch.gather(allbeam_states, 0, beam_indices)

        output_ids = generate_output.sequences
        # The "1:" is to chop off the BOS token that doesn't have a hidden state
        target_features = self._pool_decoder_states(decoder_hidden_states, output_ids[:, 1:] != self.tokenizer.pad_token_id)

        stance_feature_vec = self._forward_att(generate_output.encoder_hidden_states[-1],
                                                       target_features,
                                                       batch['attention_mask'])

        stance_logits = self.stance_classifier(stance_feature_vec)
        stance_preds = torch.argmax(stance_logits, axis=-1)

        output_texts = [self.__detokenize(id_seq) for id_seq in output_ids.detach().cpu().tolist()]
        output_embeddings = np.stack([self.fast_text.wv[text] for text in output_texts])
        output_embeddings = torch.tensor(output_embeddings).to(stance_preds.device)
        output_embeddings = output_embeddings / torch.linalg.norm(output_embeddings, keepdim=True, dim=1)

        target_scores = output_embeddings @ self.target_embeddings
        sim_scores, sim_inds = torch.max(target_scores, axis=-1)
        target_preds = torch.where(sim_scores >= self.related_threshold, sim_inds, 0)

        return TGOneShotModule.Output(
            target_preds=target_preds,
            stance_preds=stance_preds
        )

    class Encoder(Encoder):
        def __init__(self, module: TGOneShotModule):
            self.module = module
            self.tokenizer = module.tokenizer
            self.max_length = self.module.bart.config.max_position_embeddings

        def encode(self, sample: Sample, inference=False, predict_task = None):
            encoding = self.tokenizer(text=sample.context.lower(),
                                      text_target=sample.target_label.lower(),
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=self.max_length)
            encoding['stance'] = torch.tensor([sample.stance])
            encoding['stype'] = torch.tensor(sample.sample_type)
            if sample.sample_type == SampleType.SD:
                encoding['target'] = torch.tensor(
                    [self.module.targets.index(sample.target_label)],
                    dtype=torch.long)
            return encoding

        def collate(self, samples):
            encoding = collate_ids(self.tokenizer, samples, return_attention_mask=True)
            if 'target' in samples[0]:
                encoding['target'] = keyed_scalar_stack(samples, 'target')
            encoding['stance'] = keyed_scalar_stack(samples, 'stance')
            first_type = samples[0]['stype']
            assert all(s['stype'] == first_type for s in samples)
            encoding['stype'] = first_type
            return encoding

__all__ = ["OneShotModule", "ClassfierOneShotModule", "TGOneShotModule"]