from typing import Dict, Set

CANON_PROJECTS: Dict[str, Set[str]] = {
    "Ashar": {"asher", "Asher"},
    "pulse": {"pulse", "pals"},
    "axis": {"axis", "akses"},
    "metro": {"metro", "mettro"},
    "arize": {"arize", "arise", "arais", "ariez"},
    "titan": {"titan", "titaan"},
    "mapple": {"mapple", "maple", "mappel"},
    "edge": {"edge", "ej"},
    "aria": {"aria", "ariya", "aariya", "ariaa"}
}

CANON_CATEGORIES: Dict[str, Set[str]] = {
    "under_construction": {
        "under construction", "under-construction", "ongoing",
        "nirman adhin", "bhandkam shuru", "work in progress", "kaam chal raha hai"
    },
    "ready_to_move": {
        "ready to move","ready to move in","ready-to-move", "ready", "ready possession", "ready posession",
        "rehne ke liye tayyar", "tayyar ghar", "tayyar to move", "ready hai", "rtm"
    },
    "completed": {
        "completed", "delivered", "poora", "done", "handed over",
        "poora ho gaya", "complete ho gaya"
    }
}

LEX_YES = {
    "yes", "yeah", "yup", "sure", "please do", "end it", "disconnect",
    "haan", "bilkul", "ji haan", "haan ji"
}
LEX_NO = {
    "no", "nope", "not now", "wait", "continue", "carry on",
    "nahi", "nahin", "abhi nahi", "ruko", "jari rakho"
}
LEX_BYE = {
    "bye", "goodbye", "good bye", "thanks that's all", "nothing else",
    "end the call", "cut the call", "disconnect", "hang up",
    "bas itna hi", "theek hai bye", "call kat do", "disconnect kar do"
}

def apply_simple_map(text: str) -> str:
    if not text:
        return text
    t = text
    t = t.replace("maple", "mapple")
    t = t.replace("arise", "arize")
    t = t.replace("ariya", "aria").replace("aariya", "aria").replace("ariaa", "aria")
    return t
