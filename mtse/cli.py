# STL
import typing
import pathlib
# 3rd Party
from lightning.pytorch.cli import LightningCLI
# Local
from .modules import *
from .data import *
from .callbacks import *
from lightning.pytorch.callbacks import Callback

class FieldSetterCallback(Callback):
    """
    Simple callback used to set some fields that are
    easier to set programmatically than with a YAML config.
    """
    def __init__(self, dataloader_labels: Optional[List[str]] = None):
        self.dataloader_labels = dataloader_labels
    def _on_infer_start(self, trainer, pl_module):
        if self.dataloader_labels is not None:
            callbacks = trainer.callbacks
            for callback in filter(lambda c: c is not self, callbacks):
                if hasattr(callback, "dataloader_labels"):
                    callback.dataloader_labels = self.dataloader_labels

    def on_predict_start(self, trainer, pl_module):
        self._on_infer_start(trainer, pl_module)
    def on_test_start(self, trainer, pl_module):
        self._on_infer_start(trainer, pl_module)

   


class StanceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """
        parser.add_argument("--weight_ckpt", type=pathlib.Path, required=False)

    def after_instantiate_classes(self):
        model = self.model
        # Set the encoder after instantiation
        datamodule = typing.cast(BaseDataModule, self.datamodule)
        datamodule.encoder = typing.cast(BaseModule, model).encoder

        if self.config_dump.get('weight_ckpt'):
            state_dict =  torch.load(self.config_dump['weight_ckpt'])['state_dict']
            self.model.load_state_dict(state_dict, strict=False)

        extra_callback = FieldSetterCallback(datamodule.testloader_labels)
        self.trainer.callbacks.append(extra_callback)


def cli_main(**cli_kwargs):
    return StanceCLI(
        model_class=BaseModule, subclass_mode_model=True,
        datamodule_class=BaseDataModule, subclass_mode_data=True,
        trainer_defaults={
            "max_epochs": 1000,
            "deterministic": True
        },
        seed_everything_default=0,
        **cli_kwargs
    )
