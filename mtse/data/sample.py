import dataclasses
from typing import List, Optional
# Local
from .stance import BaseStance

@dataclasses.dataclass
class Sample:
    context: str 
    stance: BaseStance
    target: Optional[str] = None
    lang: Optional[str] = None


__all__ = ["Sample"]