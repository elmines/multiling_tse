# STL
import dataclasses
import pathlib
# 3rd Party
import torch
# Local
from .base_module import BaseModule
from .mixins import TargetMixin
from ..data.encoder import NoopEncoder, Encoder, keyed_scalar_stack
from ..data.transforms import SetTargetPred

class DotDict:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name not in self._data:
            raise AttributeError(f'Field "{name}" not in DotDict')
        return self._data[name]

class TargetPredModule(BaseModule):
    def __init__(self, targets_path: pathlib.Path, map_file: pathlib.Path):
        super().__init__()
        self._encoder = TargetPredModule.Encoder(targets_path, map_file)
    @property
    def encoder(self):
        return self._encoder
    def _infer_step(self, batch):
        assert isinstance(batch, dict)
        return DotDict(batch)
    class Encoder(Encoder, TargetMixin):
        def __init__(self, targets_path: pathlib.Path, map_file: pathlib.Path):
            Encoder.__init__(self)
            TargetMixin.__init__(self, targets_path)
            self.add_transform(SetTargetPred(map_file))
        def _encode(self, sample, inference=False, predict_task = None):
            target_label = sample.target_label
            target_pred = sample.target_pred
            assert target_pred is not None
            assert target_pred.gt_target == target_label
            return {
                "target": torch.tensor(self.targets.index(target_label)),
                "target_preds": torch.tensor(self.targets.index(target_pred.mapped_target)),
                "sample_id": torch.tensor(target_pred.sample_id),
            }
        def _collate(self, samples):
            return {
                k:keyed_scalar_stack(samples, k) for k in ['target', 'target_preds', 'sample_id']
            }

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

__all__ = ["TargetPredModule", "PassthroughModule"]