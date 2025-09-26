# STL
from typing import List, Dict, Optional
import functools
import enum
import abc
import copy
# 3rd party
import torch
from transformers import PreTrainedTokenizerFast
# Local
from ..types import TensorDict
from .transforms import Transform
from .sample import Sample

PoolIndices = Dict[int, List[int]]

@enum.unique
class PredictTask(enum.IntEnum):
    STANCE = 0
    TARGET = 1
    BOTH = 2

class Encoder(abc.ABC):

    def __init__(self):
        self._transforms: List[Transform] = []

    def add_transform(self, t: Transform):
        self._transforms.append(t)

    def encode(self, sample: Sample, *args, **kwargs):
        """
        DO NOT OVERRIDE THIS. Rather, override _encode
        """
        if self._transforms:
            s = copy.deepcopy(sample)
            for t in self._transforms:
                t(s)
        else:
            s = sample
        return self._encode(s, *args, **kwargs)

    @abc.abstractmethod
    def _encode(self, sample: Sample, inference=False, predict_task: Optional[PredictTask] = None) -> TensorDict:
        pass

    @abc.abstractmethod
    def collate(self, samples: List[TensorDict]) -> TensorDict:
        pass

class NoopEncoder(Encoder):
    def _encode(self, *args, **kwargs):
        return {}
    def collate(self, *args, **kwargs):
        return {}

def try_add_position_ids(encoding: TensorDict):
    if 'position_ids' not in encoding:
        encoding['position_ids'] = torch.arange(encoding['input_ids'].numel()).unsqueeze(0)

def keyed_pad(samples: List[TensorDict], k: str, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.squeeze(s[k], dim=0) for s in samples],
        batch_first=True,
        padding_value=padding_value)

def keyed_scalar_stack(samples: List[TensorDict], k: str):
    return torch.stack([torch.squeeze(s[k]) for s in samples])

def concat_lists(samples: List[TensorDict], k: str):
    return functools.reduce(lambda accum, el: accum + el, map(lambda x: x[k], samples))

def collate_ids(tokenizer: PreTrainedTokenizerFast,
                samples: List[TensorDict],
                return_attention_mask: bool = False) -> TensorDict:
    token_padding = tokenizer.pad_token_id
    rdict = {}
    rdict['input_ids'] = keyed_pad(samples, 'input_ids', padding_value=token_padding)
    if return_attention_mask:
        rdict['attention_mask'] = rdict['input_ids'] != token_padding
    if 'position_ids' in samples[0]:
        # FIXME: Need a custom pad value for this?
        rdict['position_ids'] = keyed_pad(samples, 'position_ids')
    if 'token_type_ids' in samples[0]:
        rdict['token_type_ids'] = keyed_pad(samples, 'token_type_ids', padding_value=tokenizer.pad_token_type_id)
    if 'labels' in samples[0]:
        rdict['labels'] = keyed_pad(samples, 'labels', padding_value=token_padding)
    return rdict