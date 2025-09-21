# STL
import dataclasses
# 3rd Party
import torch
# Local
from .base_module import BaseModule
from ..data.encoder import NoopEncoder

class DotDict:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name not in self._data:
            raise AttributeError(f'Field "{name}" not in DotDict')
        return self._data[name]

class PassthroughModule(BaseModule):
    """
    Simple pass-through of a CSV file of predictions.
    """

    @dataclasses.dataclass
    class Output:
        target_preds: torch.Tensor

    def __init__(self):
        super().__init__()
        self._encoder = NoopEncoder()

    @property
    def encoder(self):
        return self._encoder

    def _infer_step(self, batch):
        assert isinstance(batch, dict)
        return DotDict(batch)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__} is only for inference")
    def validation_step(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__} is only for inference")

__all__ = ["PassthroughModule"]