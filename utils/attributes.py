from typing import Optional, Dict, Set, List, Tuple
from rapidfuzz import process, fuzz

# English + Hinglish (romanized Hindi/Marathi) keyword sets per attribute
ATTR_MAP: Dict[str, Dict[str, Set[str]]] = {
    "config": {
        "en": {
            "config", "configuration",
            "bhk", "1bhk", "2bhk", "3bhk", "1 bhk", "2 bhk", "3 bhk",
            "flat type", "apartment type"
        },
        "hi": {
            "bhk", "configuration", "config",
            "1bhk", "2bhk", "3bhk", "1 bhk", "2 bhk", "3 bhk",
            "flat type", "apartment type"
        },
        "mr": {
            "bhk", "configuration", "config",
            "1bhk", "2bhk", "3bhk", "1 bhk", "2 bhk", "3 bhk",
            "flat type", "apartment type"
        },
    },

    "price": {
        "en": {
            "price", "starting price", "start price", "starting from",
            "cost", "budget", "rate", "rates", "how much", "how much is it"
        },
        "hi": {
            "keemat", "kimat", "kitna", "kitni", "kitne",
            "kya price", "price", "cost", "budget",
            "starting price", "start price", "starting from",
            "shuruaat ki keemat", "shuruat ki keemat"
        },
        "mr": {
            "kimat", "kiti", "kiti price",
            "price", "cost", "budget", "rate", "rates",
            "suruvatichi kimat", "suruvati kimat", "starting price", "start price", "starting from"
        },
    },

    "floors": {
        "en": {
            "floors", "floor", "floor count", "how many floors", "number of floors",
            "storeys", "stories"
        },
        "hi": {
            "manzil", "manzile", "manjil", "manjile",
            "kitni manzil", "kitni manjil", "kitni manzile",
            "floors", "kitni floors", "number of floors"
        },
        "mr": {
            "majle", "majla", "majlya",
            "kiti majle", "kitki majle",
            "floors", "number of floors"
        },
    },

    "towers": {
        "en": {
            "towers", "tower", "blocks", "block",
            "how many towers", "number of towers"
        },
        "hi": {
            "tower", "towers", "block", "blocks",
            "kitne tower", "kitne towers", "kitne block", "number of towers"
        },
        "mr": {
            "tower", "towers", "block", "blocks",
            "kiti tower", "kitki tower", "number of towers"
        },
    },
}

# Extra common ASR misspellings / phonetic variants
FUZZY_ALIASES: Dict[str, Set[str]] = {
    "floors": {
        "flores", "flors", "flore", "flor", "flr", "flrs", "floorz","flour","flours",
        "storey", "storie", "storiez",
    },
    "price": {
        "prise", "praice", "prize", "prys", "pries",
    },
    "towers": {
        "towrs", "twr", "twrs", "tawers", "tawrs", "tovers", "blocks tower",
    },
    "config": {
        "cnfg", "configration", "konfig", "konfiguration",
    },
}

# Build a flat search space for fuzzy matching
def _build_phrase_table() -> List[Tuple[str, str]]:
    """
    Returns a list of (attr, phrase) for all language groups + fuzzy aliases.
    """
    table: List[Tuple[str, str]] = []
    for attr, groups in ATTR_MAP.items():
        for phrases in groups.values():
            for p in phrases:
                table.append((attr, p.lower()))
    for attr, extras in FUZZY_ALIASES.items():
        for p in extras:
            table.append((attr, p.lower()))
    return table

_PHRASE_TABLE = _build_phrase_table()
_ALL_PHRASES = [p for _, p in _PHRASE_TABLE]


def detect_attribute(text: str) -> Optional[str]:
    """
    Detects which attribute the user asked for, robust to ASR typos.
    Strategy:
      1) Fast path: substring check against all phrases.
      2) Fuzzy path: WRatio against the global phrase list; accept best >= 80.
    """
    if not text:
        return None
    t = text.lower().strip()

    # 1) Substring fast-path
    for attr, phrase in _PHRASE_TABLE:
        if phrase in t:
            return attr

    # 2) Fuzzy match (handles cases like "flores")
    match = process.extractOne(t, _ALL_PHRASES, scorer=fuzz.WRatio)
    if match:
        phrase, score, idx = match
        if score >= 80:
            return _PHRASE_TABLE[idx][0]

    return None
