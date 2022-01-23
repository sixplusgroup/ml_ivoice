import sys
import unicodedata
from plane import build_new_regex

NORMAL_TOKEN_TAG = "O"
EMAIL_TOKEN = "<EMAIL>"
URL_TOKEN = "<URL>"
TELEPHONE_TOKEN = "<TEL>"
CURRENCY_TOKEN = "<CURRENCY>"
NUMBER_TOKEN = "<NUM>"

URL = build_new_regex(
    "url_checking",
    r"https?:\/\/[!-~]+|[a-zA-Z0-9_.+-]+\.[a-zA-Z0-9-]+$",
)

currency_list = "|".join(
    [
        chr(c)
        for c in range(sys.maxunicode)
        if unicodedata.category(chr(c)).startswith(("Sc"))
    ]
)

CURRENCY = build_new_regex(
    "currency", r"(\{})\d+([.,]?\d*)*([A-Za-z]+)?".format(currency_list)
)

NUMBER = build_new_regex("number", r"[0-9]*[.]?[0-9]+[%]?")

DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP = {
    NORMAL_TOKEN_TAG: ("", False),
    "C_COMMA": ("，", False),
    "C_PERIOD": ("。", True),
    "C_QUESTIONMARK": ("？", True),
    "C_EXLAMATIONMARK": ("！", True),
    "C_COLON": ("：", True),
    "C_DUNHAO": ("、", False),
}

DEFAULT_CHINESE_NER_MAPPING = {
    "，": "C_COMMA",
    "。": "C_PERIOD",
    "？": "C_QUESTIONMARK",
    "！": "C_EXLAMATIONMARK",
    "：": "C_COLON",
    "、": "C_DUNHAO",
}

ALL_PUNCS = [
    c
    for c in range(sys.maxunicode)
    if unicodedata.category(chr(c)).startswith(("P", "Cc"))
]