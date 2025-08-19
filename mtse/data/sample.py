import dataclasses
from typing import Optional
import enum
# Local
from .stance import BaseStance

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

    target_pred: Optional[str] = None
    """
    What some previous component in the pipeline
    predicted as the target for the context
    """

    target_label: Optional[str] = None
    """
    The ground truth target for the context
    """

    target_input: Optional[str] = None
    """
    What target we want to actually tokenize
    along with the context
    """

    lang: Optional[str] = None

    sample_type: SampleType = SampleType.SD


__all__ = ["Sample", "SampleType"]