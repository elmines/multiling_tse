# STL
import typing
import pathlib
# 3rd Party
from lightning.pytorch.cli import LightningCLI
# Local
from .modules import *
from .data import *
from .callbacks import *

class StanceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """
        parser.add_argument("--weight_ckpt", type=pathlib.Path, required=False)

    def after_instantiate_classes(self):
        model = self.model
        typing.cast(BaseDataModule, self.datamodule).encoder = typing.cast(BaseModule, model).encoder
        if self.config_dump.get('weight_ckpt'):
            state_dict =  torch.load(self.config_dump['weight_ckpt'])['state_dict']
            self.model.load_state_dict(state_dict, strict=False)

def cli_main(**cli_kwargs):
    return StanceCLI(
        model_class=BaseModule, subclass_mode_model=True,
        datamodule_class=BaseDataModule, subclass_mode_data=True,
        trainer_defaults={
            "max_epochs": 1000,
            "deterministic": True
        },
        seed_everything_default=1,
        **cli_kwargs
    )
