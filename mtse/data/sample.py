import dataclasses
from typing import Optional
import enum
import pathlib
# Local
from .stance import BaseStance
from .target_pred import TargetPred

class SampleType(enum.IntEnum):
    SD = 0
    """
    Stance detection
    """

    KG = 1
    """
    Keyword generation
    """

@dataclasses.dataclass
class Sample:
    context: str 
    stance: BaseStance
    source_path: pathlib.Path # Need the provenance of this sample somehow

    target_pred: Optional[TargetPred] = None
    """
    Predicted target for the context
    """

    target_label: Optional[str] = None
    """
    The ground truth target for the context
    """

    lang: Optional[str] = None

    sample_type: SampleType = SampleType.SD

    _target_input: Optional[str] = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        self._target_input = None

    @property
    def target_input(self) -> Optional[str]:
        """
        What target we want to actually tokenize
        along with the context.
        """
        if self._target_input is not None:
            return self._target_input
        return self.target_label

    @target_input.setter
    def target_input(self, v: str):
        self._target_input = v


__all__ = ["Sample", "SampleType"]