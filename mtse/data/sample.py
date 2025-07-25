import dataclasses
from typing import List, Optional
# Local
from .stance import BaseStance

@dataclasses.dataclass
class Sample:
    context: str 
    target: str
    stance: BaseStance
    lang: Optional[str] = None


__all__ = ["Sample"]