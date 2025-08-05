import abc
import json
import re
from typing import Dict, Optional
import importlib.resources
# 3rd Party
import wordninja
import preprocessor as twp
twp.set_options(twp.OPT.URL, twp.OPT.EMOJI, twp.OPT.RESERVED)
# Local
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

class LiKeywordRemoval(Transform):

    # Replacement dict is from Li et al. (2023)'s work
    KEYWORDS =  [
        'Joe',
        'Biden',
        'Bernie',
        'Sanders',
        'Donald',
        'Trump',
        'abortion',
        'cloning',
        'death',
        'penalty',
        'gun',
        'control',
        'marijuana',
        'legalization',
        'minimum',
        'wage',
        'nuclear',
        'energy',
        'school',
        'uniforms',
        'Atheism',
        'Feminist',
        'Movement',
        'Hillary',
        'Clinton',
        'face',
        'masks',
        'fauci',
        'stay',
        'home',
        'school',
        'closures',
        'orders'
    ]

    KEYWORD_PATT = re.compile('|'.join(KEYWORDS), flags=re.IGNORECASE)

    def __call__(self, sample: Sample):
        old_context = sample.context
        new_context, count = LiKeywordRemoval.KEYWORD_PATT.subn('', old_context)
        sample.context = new_context

class LiPreprocess(Transform):

    KEYWORD_DICT: Optional[Dict[str, str]] = None

    PHRASE_PATTERN = re.compile(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+")

    def __init__(self):
        self._keyword_dict = LiPreprocess.get_keyword_dict()

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
        old_context = sample.context

        twp_cleaned = twp.clean(old_context)

        converted = []
        phrases = LiPreprocess.PHRASE_PATTERN.findall(twp_cleaned)
        for phrase in phrases:
            lowered = phrase.lower()
            if lowered in self._keyword_dict:
                conversion = [self._keyword_dict[lowered]]
                converted.append(conversion)
            elif phrase.startswith('#') or phrase.startswith('@'):
                conversion = wordninja.split(phrase)
                converted.append(conversion)
            else:
                converted.append([phrase])
        new_context = " ".join(tok for tok_set in converted for tok in tok_set)
        sample.context = new_context


__all__ = [
    "Transform",
    "SemHashtagRemoval",
    "LiKeywordRemoval",
    "LiPreprocess",
]