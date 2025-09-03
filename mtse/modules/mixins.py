import pathlib
from ..constants import UNRELATED_TARGET

class TargetMixin:
    """
    Mixin for lightning modules that need to keep a list of possible
    targets
    """

    def __init__(self, targets_path: pathlib.Path):
        super().__init__()
        self.no_target = 0
        with open(targets_path, 'r') as r:
            targets = [t.strip() for t in r]
        self.targets = [UNRELATED_TARGET] + targets
    
    @property
    def n_targets(self):
        return len(self.targets)
