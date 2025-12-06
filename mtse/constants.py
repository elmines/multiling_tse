DEFAULT_BATCH_SIZE = 32

DEFAULT_HF_MODEL = "vinai/bertweet-base"

UNRELATED_TARGET = "Unrelated"

INDEPENDENCE_TARGETS = [
    "Catalonian Independence",
    "Sardinian Independence"
]
INDEPENDENCE = "Independence"

TARGET_DELIMITER = ";"

DEFAULT_RELATED_THRESHOLD = 0.2

# TODO: Get this from a config file instead?
LANGS = [
    'ca',
    'cs',
    'en',
    'es',
    'et',
    'fr',
    'it',
    'hi',
    'zh'
]
ID_TO_LANG = dict(enumerate(LANGS))
LANG_TO_ID = {v:k for k,v in ID_TO_LANG.items()}

# Column Names

C_SAMPLE = "Sample"

C_UNTRANSLATED_TARGET = "UntranslatedTarget"

C_GENERATED_TARGET = "GeneratedTarget"

C_MAPPED_TARGET = "MappedTarget"

C_PRED_STANCE = "StancePred"

C_GT_TARGET = "Target"
C_GT_STANCE = "Stance"

C_LANG = "Lang"