import abc
import copy
import re

from .sample import Sample


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        pass

class SemHashtagRemoval(Transform):
    def __init__(self):
        self.pattern = re.compile('#SemST', flags=re.IGNORECASE)
    def __call__(self, sample: Sample) -> Sample:
        s = copy.deepcopy(sample)
        s.context = self.pattern.sub('', s.context)
        return s

class LiKeywordRemoval(Transform):

    def __init__(self):
        # Replacement dict is from Li et al. (2023)'s work
        replace_strings = {
            'PStance'      : ['Joe', 'Biden', 'Bernie', 'Sanders', 'Donald', 'Trump'],
            'AM'           : ['abortion', 'cloning', 'death', 'penalty', 'gun', 'control', 'marijuana', 'legalization', 'minimum', 'wage', 'nuclear', 'energy', 'school', 'uniforms'],
            'SemEval2016'  : ['Atheism', 'Feminist', 'Movement', 'Hillary',  'Clinton', 'Legalization', 'Abortion'],
            'Covid19'      : ['face', 'masks', 'fauci', 'stay', 'home', 'orders', 'school', 'closures'],
            'Stance_Merge_Unrelated': ['Joe', 'Biden', 'Bernie', 'Sanders', 'Donald', 'Trump', 'abortion', 'cloning', 'death', 'penalty', 'gun', 'control', 'marijuana', 'legalization', 'minimum', 'wage',  'nuclear', 'energy', 'school', 'uniforms', 'Atheism', 'Feminist', 'Movement', 'Hillary', 'Clinton', 'face', 'masks', 'fauci', 'stay', 'home', 'school', 'closures', 'orders']
        }
        merged_keywords = {k.lower() for keyword_set in replace_strings.values() for k in keyword_set}
        self.pattern = re.compile('|'.join(merged_keywords), flags=re.IGNORECASE)

    def __call__(self, sample: Sample):
        s = copy.deepcopy(sample)
        s.context = self.pattern.sub('', s.context)
        return s

__all__ = ["Transform", "SemHashtagRemoval", "LiKeywordRemoval"]