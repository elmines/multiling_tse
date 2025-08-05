import abc
import json
import re
from typing import Dict, Optional
import importlib.resources
from .sample import Sample


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sample: Sample) -> None:
        """
        Modifies the sample in-place
        """

class SemHashtagRemoval(Transform):
    def __init__(self):
        self.pattern = re.compile('#SemST', flags=re.IGNORECASE)
    def __call__(self, sample: Sample):
        sample.context = self.pattern.sub('', sample.context)

class LiPretokenize(Transform):
    def __init__(self):
        self.pattern = re.compile(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+")
    def __call__(self, sample: Sample):
        old_context = sample.context
        new_context = " ".join(self.pattern.findall(old_context))
        sample.context = new_context

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
        sample.context = self.pattern.sub('', sample.context)

class LiSlangExpansion(Transform):

    KEYWORD_DICT: Optional[Dict[str, str]] = None

    def __init__(self):
        self._keyword_dict = LiSlangExpansion.get_keyword_dict()

    @classmethod
    def get_keyword_dict(cls) -> Dict[str, str]:
        if cls.KEYWORD_DICT is None:
            res_files = importlib.resources.files('mtse.res')
            emnlp_text = res_files.joinpath('emnlp_dict.txt').read_text()
            d = {}
            for l in emnlp_text.replace('\r', '').strip().split('\n'):
                pair = l.strip().split()
                d[pair[0]] = pair[1]

            slang_json = json.loads(res_files.joinpath('noslang_data.json').read_text())
            d.update(slang_json)

            cls.KEYWORD_DICT = d
        return cls.KEYWORD_DICT

    def __call__(self, sample: Sample):
        converted = []
        hits = []
        for tok in sample.context.split():
            lowered = tok.lower()
            if lowered in self._keyword_dict:
                conversion = self._keyword_dict[lowered]
                converted.append(conversion)
                hits.append((tok, conversion))
            else:
                converted.append(tok)
        sample.context = " ".join(converted)


__all__ = [
    "Transform",
    "SemHashtagRemoval",
    "LiPretokenize",
    "LiKeywordRemoval",
    "LiSlangExpansion"
]