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

_semeval_tag = re.compile('#SemST', flags=re.IGNORECASE)
def _remove_semeval_tag(text):
    return _semeval_tag.sub('', text)

class SemHashtagRemoval(Transform):
    def __call__(self, sample: Sample):
        sample.context = _remove_semeval_tag(sample.context)

class LiPreprocess(Transform):
    """
    Performs the preprocessing logic from Li et al. (2023)'s work
    """

    TARGET_WORDS =  [
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

    TARGET_PATT = re.compile('|'.join(TARGET_WORDS), flags=re.IGNORECASE)

    KEYWORD_DICT: Optional[Dict[str, str]] = None

    PHRASE_PATTERN = re.compile(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+")

    def __init__(self,
                 scrub_targets: bool = False,
                 remove_se_hashtag: bool = True):
        self._keyword_dict = LiPreprocess.get_keyword_dict()
        self.scrub_targets = scrub_targets
        self.remove_se_hashtag = remove_se_hashtag

    @classmethod
    def get_keyword_dict(cls) -> Dict[str, str]:
        if cls.KEYWORD_DICT is None:
            res_files = importlib.resources.files('mtse.res')
            emnlp_text = res_files.joinpath('emnlp_dict.txt').read_text()
            emnlp_dict = {}
            for l in emnlp_text.replace('\r', '').strip().split('\n'):
                pair = l.strip().split()
                emnlp_dict[pair[0]] = pair[1]

            slang_json = json.loads(res_files.joinpath('noslang_data.json').read_text())

            # The order of these 2 matters, because 253 keys
            # are in both dictionaries. emnlp's takes precedence
            cls.KEYWORD_DICT = {**slang_json, **emnlp_dict}
        return cls.KEYWORD_DICT

    def _clean_text(self, context):
        # 3. Use tweet-preprocessor
        context = twp.clean(context)
        # 4. Remove SemEval hashtags
        if self.remove_se_hashtag:
            context = _remove_semeval_tag(context)
        # 5. Normalize slang and split hashtags/mentions
        converted = []
        for phrase in LiPreprocess.PHRASE_PATTERN.findall(context):
            lowered = phrase.lower()
            if lowered in self._keyword_dict:
                # The keyword dict actually has some uppercase words,
                # meaning we'll have some uppercase letters in the final context.
                # But that's what Li did in his code
                conversion = [self._keyword_dict[lowered]]
                converted.append(conversion)
            elif phrase.startswith('#') or phrase.startswith('@'):
                conversion = wordninja.split(phrase)
                converted.append(conversion)
            else:
                converted.append([phrase])
        context = " ".join(tok for tok_set in converted for tok in tok_set)
        return context


    def __call__(self, sample: Sample):
        context = sample.context
        # 1. Lowercase the text (even though he was using a case-sensitive tokenizer)
        context = context.lower()
        # 2. Remove target keywords
        if self.scrub_targets:
            context = LiPreprocess.TARGET_PATT.sub('', context)
        context = self._clean_text(context)
        sample.context = context

        if sample.target_input is not None:
            target_input = sample.target_input
            target_input = target_input.lower()
            target_input = self._clean_text(target_input)
            sample.target_input = target_input


__all__ = [
    "Transform",
    "SemHashtagRemoval",
    "LiPreprocess",
]