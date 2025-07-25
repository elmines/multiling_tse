# STL
import abc
# 3rd Party
import lightning as L
import torch
# Local
from ..data import Encoder

class BaseModule(L.LightningModule):
    """
    Dummy module to restrict what modules can be given at the CLI
    """

    def validation_step(self, batch, batch_idx):
        return self._infer_step(batch)
    def test_step(self, batch, batch_idx):
        return self._infer_step(batch)
    @abc.abstractmethod
    def _infer_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=4e-5)
    def on_train_epoch_start(self):
        self.train()
    def on_validation_epoch_start(self):
        self.eval()
    def on_test_epoch_start(self):
        self.eval()

    @property
    @abc.abstractmethod
    def encoder(self) -> Encoder:
        raise NotImplementedError

