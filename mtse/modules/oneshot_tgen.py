# STL
from __future__ import annotations
import pathlib
import dataclasses
import re
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, BartTokenizerFast
from gensim.models import FastText
# Local 
from .mixins import TargetMixin
from .base_module import BaseModule
from ..data import Encoder, StanceType, STANCE_TYPE_MAP, Sample, collate_ids, keyed_scalar_stack, SampleType
from ..constants import DEFAULT_RELATED_THRESHOLD
from ..mapping import make_target_embeddings, detokenize_generated_targets, map_targets

class TGOneShotModule(BaseModule, TargetMixin):

    @dataclasses.dataclass
    class TrainOutput:
        lm_loss: torch.Tensor
        stance_loss: torch.Tensor

    @dataclasses.dataclass
    class InferOutput:
        target_preds: torch.Tensor
        stance_preds: torch.Tensor

    DEFAULT_PRETRAINED_MODEL = "facebook/bart-base"

    EXCLUDE_KWARGS = {'target', 'stance', 'stype', 'lm_weight'}

    __POSTPROC_PATT_1 = re.compile(r'[;,\.]|\s')
    __POSTPROC_PATT_2 = re.compile(r'  +')

    @classmethod
    def _postprocess(cls, keyphrase):
        k1 = cls.__POSTPROC_PATT_1.sub(' ', keyphrase)
        k2 = cls.__POSTPROC_PATT_2.sub(' ', k1)
        return k2

    def __init__(self,
                 embeddings_path: pathlib.Path,
                 targets_path: pathlib.Path,
                 stance_type: StanceType,
                 related_threshold: float = DEFAULT_RELATED_THRESHOLD,
                 backbone_lr: float = 1e-5,
                 head_lr: float = 4e-5,
                 max_length: int = 75,
                 fixed_lm: bool = False,
                 pretrained_model: str = DEFAULT_PRETRAINED_MODEL,
                 use_target_gt: bool = False,
                 **parent_kwargs):
        BaseModule.__init__(self, **parent_kwargs)
        TargetMixin.__init__(self, targets_path)

        self.stance_type = STANCE_TYPE_MAP[stance_type]
        self.related_threshold = related_threshold
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.max_length = max_length
        self.fixed_lm = fixed_lm
        self.use_target_gt = use_target_gt

        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer: PreTrainedTokenizerFast = BartTokenizerFast.from_pretrained(pretrained_model, normalization=True)

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
        self.target_embeddings: torch.Tensor # Only have this so the IDE knows this variable exists
        self.register_buffer("target_embeddings", torch.tensor(make_target_embeddings(self.targets, self.fast_text)), persistent=False)

        self.__encoder = self.Encoder(self)

        if self.fixed_lm:
            for name, param in self.bart.named_parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        param_groups = [
            {"params": self.stance_classifier.parameters(), "lr": self.head_lr},
            {"params": self.cross_att.parameters(), "lr": self.head_lr},
        ]
        if not self.fixed_lm:
            lm_head_params = set(self.bart.lm_head.parameters())
            # One overlapping parameter forces me to do this
            backbone_params = [p for p in self.bart.model.parameters() if p not in lm_head_params]
            lm_head_params = list(lm_head_params)
            param_groups.append({"params": lm_head_params, "lr": self.backbone_lr})
            param_groups.append({"params": backbone_params, "lr": self.head_lr})
        return torch.optim.AdamW(param_groups)

    @property
    def encoder(self):
        return self.__encoder


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
                                             max_length=self.max_length,
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

        if self.use_target_gt:
            target_preds = batch['target']
        else:
            all_texts, sample_inds = detokenize_generated_targets(generate_output, self.tokenizer)
            sample_inds = torch.tensor(sample_inds, device=self.device)
            target_preds, _ = map_targets(
                self.fast_text,
                self.target_embeddings,
                all_texts,
                sample_inds,
                self.related_threshold
            )
        return TGOneShotModule.InferOutput(
            target_preds=target_preds,
            stance_preds=stance_preds
        )

    class Encoder(Encoder):
        def __init__(self, module: TGOneShotModule):
            super().__init__()
            self.module = module
            self.tokenizer = module.tokenizer
            self.max_length = self.module.bart.config.max_position_embeddings

        def _encode(self, sample: Sample, inference=False, predict_task = None):
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

__all__ = ["TGOneShotModule"]