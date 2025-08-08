# 3rd Party
from lightning.pytorch.cli import LightningCLI
# Local
from .modules import *
from .data import *
from .callbacks import TSEStatsCallback, TargetClassificationStatsCallback, TargetPredictionWriter, StanceClassificationStatsCallback

class StanceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """

    def after_instantiate_classes(self):
        model = self.model
        self.datamodule.encoder = model.encoder
        if isinstance(self.model, OneShotModule):
            pass
        elif isinstance(self.model, TargetModule):
            # TODO: Just make the uesr specify these callbacks in the YAML config
            self.trainer.callbacks.append(TargetClassificationStatsCallback(len(self.model.targets) + 1))
            self.trainer.callbacks.append(TargetPredictionWriter(self.trainer.logger.log_dir))
        elif isinstance(self.model, TwoShotModule):
            pass
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
