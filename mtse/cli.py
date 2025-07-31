# 3rd Party
import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import pathlib
# Local
from .modules import *
from .data import *
from .callbacks import TSEStatsCallback, TargetClassificationStatsCallback

class StanceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """
        # parser.add_argument("--predict.out", type=pathlib.Path, help="Path to write predictions")

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs):
        super()._add_subcommands(parser, **kwargs)
        predict_parser = self._subcommand_parsers['predict']
        predict_parser.add_argument('--predictions', type=pathlib.Path, help='Path to write predictions')

    def after_instantiate_classes(self):
        model = self.model
        self.datamodule.encoder = model.encoder
        if isinstance(self.model, OneShotModule):
            self.trainer.callbacks.append(TSEStatsCallback())
        elif isinstance(self.model, TargetModule):
            self.trainer.callbacks.append(TargetClassificationStatsCallback(len(self.model.targets) + 1))
        elif isinstance(self.model, StanceOnlyModule):
            self.trainer.callbacks.append(TSEStatsCallback())
        else:
            raise ValueError(f"Unknown module type {type(self.model)}")

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
