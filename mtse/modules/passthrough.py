# STL
from __future__ import annotations
import dataclasses
import pathlib
# 3rd Party
import torch
# Local
from .base_module import BaseModule
from .mixins import TargetMixin
from ..data.encoder import NoopEncoder, Encoder, keyed_scalar_stack, concat_lists
from ..data.transforms import SetTargetPred
from ..callbacks.target_prediction_callback import TargetLevel
from ..constants import LANG_TO_ID

class DotDict:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name not in self._data:
            raise AttributeError(f'Field "{name}" not in DotDict')
        return self._data[name]

class TargetPredModule(BaseModule, TargetMixin):
    def __init__(self,
                 targets_path: pathlib.Path,
                 map_file: pathlib.Path,
                 input_target_level: TargetLevel = TargetLevel.mapped,
                 with_lang: bool = False
                 ):
        super().__init__()
        TargetMixin.__init__(self, targets_path)
        self.with_lang = with_lang
        self.input_target_level = input_target_level

        self._encoder = TargetPredModule.Encoder(self)
        self._encoder.add_transform(SetTargetPred(map_file))
    @property
    def encoder(self):
        return self._encoder
    def _infer_step(self, batch):
        assert isinstance(batch, dict)
        return DotDict(batch)
    class Encoder(Encoder):
        def __init__(self, module: TargetPredModule):
            Encoder.__init__(self)
            self.module = module
            assert self.module.input_target_level > TargetLevel.none

        def _encode(self, sample, inference=False, predict_task = None):
            target_pred = sample.target_pred
            assert target_pred is not None
            assert target_pred.gt_target == sample.target_label

            rdict = {
                "target": torch.tensor(self.module.targets.index(sample.target_label)),
            }
            if self.module.with_lang:
                rdict['lang'] = torch.tensor([LANG_TO_ID[target_pred.lang]], dtype=torch.long)

            if self.module.input_target_level == TargetLevel.mapped:
                rdict["sample_inds"] = torch.tensor([target_pred.sample_id])
                rdict["target_preds"] = torch.tensor(self.module.targets.index(target_pred.mapped_target))
            else:
                gen_targets = target_pred.generated_targets
                rdict["sample_inds"] = torch.full((len(gen_targets),), target_pred.sample_id)
                rdict['target_gens'] = target_pred.generated_targets
                rdict['target_untrans'] = target_pred.untranslated_targets
            return rdict

        def _collate(self, samples):
            rdict = {
                'target': keyed_scalar_stack(samples, 'target'),
                'sample_inds': torch.concatenate([s['sample_inds'] for s in samples])
            }
            if self.module.with_lang:
                rdict['lang'] = keyed_scalar_stack(samples, 'lang')
            if self.module.input_target_level != TargetLevel.mapped:
                for k in filter(lambda k: k in samples[0], ['target_untrans', 'target_gens']):
                    rdict[k] = concat_lists(samples, k)
            else:
                rdict['target_preds'] = keyed_scalar_stack(samples,'target_preds')
            return rdict

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