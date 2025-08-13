# utils/dialogue.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import os
import re
import json

# ---- Entity / attribute detectors from your project ----
try:
    from utils.entity_fuzzy import detect_project, detect_category
except Exception:
    def detect_project(text: str) -> Optional[str]:
        return None
    def detect_category(text: str) -> Optional[str]:
        return None

try:
    from utils.attributes import detect_attribute
except Exception:
    def detect_attribute(text: str) -> Optional[str]:
        # must return one of: "price", "config", "floors", "towers", or None
        t = text.lower()
        if any(k in t for k in ["price", "budget", "rate", "₹", "rs", "कीमत", "लागत", "दाम"]):
            return "price"
        if any(k in t for k in ["config", "bhk", "कॉन्फ", "बीएचके"]):
            return "config"
        if any(k in t for k in ["floor", "floors", "मंजिल", "मंज़िल"]):
            return "floors"
        if any(k in t for k in ["tower", "towers", "टावर", "टावर्स", "ब्लॉक"]):
            return "towers"
        return None

# ---- Optional classifier to keep it NON rule-only ----
def _load_classifier(intents_path: str):
    try:
        from utils.intent_classifier import SBERTIntentClassifier
        clf = SBERTIntentClassifier()
        if hasattr(clf, "load_intents"):
            clf.load_intents(intents_path)
        if hasattr(clf, "fit"):
            clf.fit()
        return clf
    except Exception:
        return None

def _predict_intent(clf, text: str, threshold: float = 0.55) -> Tuple[str, float]:
    if not clf:
        return ("fallback", 0.0)
    try:
        if hasattr(clf, "predict"):
            try:
                out = clf.predict(text, threshold=threshold)
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    return (str(out[0]), float(out[1]))
                return (str(out), 1.0)
            except TypeError:
                lab, sc = clf.predict(text)
                return (str(lab), float(sc))
    except Exception:
        pass
    return ("fallback", 0.0)

# ---- Resolve and load data/intents.json robustly ----
def _resolve_intents_path() -> str:
    try:
        import config as CFG
        p = getattr(CFG, "INTENTS_PATH", None)
        if p and os.path.exists(p):
            return os.path.abspath(p)
    except Exception:
        pass

    envp = os.environ.get("INTENTS_PATH")
    if envp and os.path.exists(envp):
        return os.path.abspath(envp)

    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(os.getcwd(), "data", "intents.json"),
        os.path.join(here, "..", "data", "intents.json"),
        os.path.join(os.getcwd(), "intents.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    raise FileNotFoundError("intents.json not found via config/ENV/fallbacks.")

_INTENTS_PATH = _resolve_intents_path()
with open(_INTENTS_PATH, "r", encoding="utf-8") as f:
    _INTENTS_RAW = json.load(f)

# Domain knowledge
_FACTS: Dict[str, dict] = _INTENTS_RAW.get("project_facts", {})
_CATS:  Dict[str, List[str]] = _INTENTS_RAW.get("project_categories", {})
_CFG:   Dict[str, str] = _INTENTS_RAW.get("config", {})
_PHONE: str = _CFG.get("phone_number", "9999999999")

# Optional classifier
_CLF = _load_classifier(_INTENTS_PATH)

# ---- Dialogue state ----
@dataclass
class DialogueCtx:
    tenant: str = "ashar"
    project: Optional[str] = None          # canonical project key (e.g., "aria")
    category: Optional[str] = None         # ready_to_move | under_construction | completed
    attribute: Optional[str] = None        # price | config | floors | towers
    last_intent: Optional[str] = None
    lang: str = "en"
    greeted: bool = False
    handoff: bool = False

# ---- Copy/text templates (Female 1st-person Hindi; 2nd-person polite-masculine) ----
T = {
    "greet": {
        "en": ("Welcome to Ashar Group. I can help with Under-Construction, "
               "Ready-to-Move, or Completed projects, and details like "
               "Configuration, Starting price, Floors and Towers. What would you like to know?"),
        "hi": ("नमस्ते! Ashar Group में आपका स्वागत है। मैं अंडर-कंस्ट्रक्शन, "
               "रेडी-टू-मूव और कम्प्लीटेड प्रोजेक्ट्स की जानकारी दे सकती हूँ — "
               "कॉन्फ़िगरेशन, स्टार्टिंग प्राइस, फ्लोर्स और टावर्स सहित। आप क्या जानना चाहेंगे?")
    },
    "ask_category": {
        "en": "Would you like Under-Construction, Ready-to-Move, or Completed?",
        "hi": "क्या आप अंडर-कंस्ट्रक्शन, रेडी-टू-मूव, या कम्प्लीटेड देखना चाहेंगे?"
    },
    "list_projects": {
        "en": "Here are the {cat} projects: {items}. Which one would you like?",
        "hi": "{cat} में ये प्रोजेक्ट्स हैं: {items}। आप कौन-सा देखना चाहेंगे?"
    },
    "proj_details": {
        "en": "{name} — Configuration: {config}; Starting from: {price}; Floors: {floors}; No. of Towers: {towers}.",
        "hi": "{name} — कॉन्फ़िगरेशन: {config}; स्टार्टिंग फ्रॉम: {price}; फ्लोर्स: {floors}; टावर्स: {towers}."
    },
    "ask_attribute": {
        "en": "Do you want price, configuration (BHK), floors, or towers?",
        "hi": "क्या आप प्राइस, कॉन्फ़िगरेशन (BHK), फ्लोर्स या टावर्स जानना चाहेंगे?"
    },
    "attr_answer": {
        "en": "{name} — {label}: {value}",
        "hi": "{name} — {label}: {value}"
    },
    "ask_project_for_attr": {
        "en": "For {label}, please pick a project in {cat}: {items}.",
        "hi": "{label} बताने के लिए कृपया {cat} में कोई प्रोजेक्ट चुनिए: {items}."
    },
    "handoff": {
        "en": "Okay, I’ll connect you to a representative now.",
        "hi": "ठीक है, मैं अभी आपको प्रतिनिधि से जोड़ रही हूँ।"
    },
    "whatsapp": {
        "en": f"Please send ‘Hi’ on WhatsApp to {_PHONE}. I’ll share the project details there.",
        "hi": f"कृपया WhatsApp पर ‘Hi’ भेजिए: {_PHONE}। मैं वहाँ विवरण साझा कर दूँगी।"
    },
    "fallback": {
        "en": "I can help with Ashar’s projects and details (Configuration, Starting price, Floors, Towers). What would you like to know?",
        "hi": "मैं Ashar के प्रोजेक्ट्स और विवरण (कॉन्फ़िगरेशन, स्टार्टिंग प्राइस, फ्लोर्स, टावर्स) में मदद कर सकती हूँ। आप क्या जानना चाहेंगे?"
    }
}

CAT_LABELS = {
    "under_construction": {"en": "Under-Construction", "hi": "अंडर-कंस्ट्रक्शन"},
    "ready_to_move":      {"en": "Ready-to-Move",      "hi": "रेडी-टू-मूव"},
    "completed":          {"en": "Completed",          "hi": "कम्प्लीटेड"},
}

ATTR_LABELS = {
    "price":  {"en": "Starting from", "hi": "स्टार्टिंग फ्रॉम"},
    "config": {"en": "Configuration", "hi": "कॉन्फ़िगरेशन"},
    "floors": {"en": "Floors",        "hi": "फ्लोर्स"},
    "towers": {"en": "No. of Towers", "hi": "टावर्स"},
}

# ---- Helpers ----
def _L(lang: str) -> str:
    return "hi" if lang == "hi" else "en"

def _pretty_list(keys: List[str]) -> str:
    names = []
    for k in keys or []:
        rec = _FACTS.get(k, {})
        names.append(rec.get("name", k.title()))
    return ", ".join(names) if names else "—"

def _category_for_project(pkey: Optional[str]) -> Optional[str]:
    if not pkey:
        return None
    rec = _FACTS.get(pkey, {})
    return rec.get("category")

def _project_answer_all(pkey: str, L: str) -> str:
    rec = _FACTS.get(pkey, {})
    return T["proj_details"][L].format(
        name=rec.get("name", pkey.title()),
        config=rec.get("config", "—"),
        price=rec.get("price", "—"),
        floors=rec.get("floors", "—"),
        towers=rec.get("towers", "—"),
    )

def _project_answer_attr(pkey: str, attr: str, L: str) -> Optional[str]:
    rec = _FACTS.get(pkey, {})
    if not rec:
        return None
    label = ATTR_LABELS.get(attr, {}).get(L, attr.title())
    value = rec.get(attr)
    if value in (None, ""):
        return None
    name = rec.get("name", pkey.title())
    return T["attr_answer"][L].format(name=name, label=label, value=value)

# Minimal rules fallback + navigation/back
_RULES = [
    (r"\bwhatsapp|व्हाट्सऐप\b", "whatsapp_details"),
    (r"\btransfer\b|representative|human|agent|कनेक्ट|प्रतिनिधि|ह्यूमन", "connect_representative"),
    (r"\b(back|go back|previous|list again|show (all )?projects)\b|वापस|पीछे|फिर से\s*लिस्ट", "go_back"),
    (r"ready\s*to\s*move|रेडी.?ट.?ू.?मूव", "ask_projects"),
    (r"under\s*construction|अंडर.?कंस्ट्रक्शन", "ask_projects"),
    (r"completed|कम्प्लीटेड|पूर्ण|डिलीवर", "ask_projects"),
    (r"\bhi\b|hello|hey|नमस्ते|हेलो|सलाम", "greet"),
]

def _rule_intent(text: str) -> Optional[str]:
    for pat, lab in _RULES:
        if re.search(pat, text, re.I):
            return lab
    return None

# ---- Public API ----
def nlu_router(text_norm: str, lang: str, ctx: DialogueCtx) -> Tuple[str, DialogueCtx]:
    """
    Hybrid router with navigation:
      • Classifier first, rules as fallback
      • Saying a category or 'projects' clears selected project (go up a level)
      • 'Back' intent navigates up one level
      • Attribute is RESET when switching project or category (precise change)
    """
    L = _L(lang)
    ctx.lang = L

    # Detect entities for THIS utterance (before mutating ctx)
    detected_proj = detect_project(text_norm)        # e.g., "metro"
    detected_cat  = detect_category(text_norm)       # e.g., "under_construction"
    detected_attr = detect_attribute(text_norm)      # e.g., "price"

    # Intent via classifier, fallback to rules
    intent, score = _predict_intent(_CLF, text_norm, threshold=0.55)
    if intent == "fallback":
        intent = _rule_intent(text_norm) or "fallback"
    ctx.last_intent = intent

    # ---- NAVIGATION: handle "back"
    if intent == "go_back":
        if ctx.project:
            # go back from project -> category list
            ctx.project = None
            # (intentionally NOT resetting attribute here unless you want that behavior too)
            if ctx.category:
                items = _CATS.get(ctx.category, [])
                if items:
                    cat_name = CAT_LABELS[ctx.category][L]
                    return T["list_projects"][L].format(cat=cat_name, items=_pretty_list(items)), ctx
                return T["ask_attribute"][L], ctx
            # if no category, go to top
            return T["ask_category"][L], ctx
        # already at category or top
        if ctx.category:
            items = _CATS.get(ctx.category, [])
            if items:
                cat_name = CAT_LABELS[ctx.category][L]
                return T["list_projects"][L].format(cat=cat_name, items=_pretty_list(items)), ctx
        return T["ask_category"][L], ctx

    # ------------------ SURGICAL CHANGE START ------------------
    # Remember previous selections to see if the user changed layer
    prev_proj = ctx.project
    prev_cat  = ctx.category

    # Update ctx with **newly detected** entities
    if detected_proj:
        # if user picked a DIFFERENT project, reset attribute
        if detected_proj != prev_proj:
            ctx.attribute = None
        ctx.project = detected_proj
        # derive category if missing
        ctx.category = _category_for_project(detected_proj) or ctx.category

    if detected_cat:
        # if user picked a DIFFERENT category, reset attribute
        if detected_cat != prev_cat:
            ctx.attribute = None
        ctx.category = detected_cat
        ctx.project = None  # moving up a level clears project

    if detected_attr:
        ctx.attribute = detected_attr
    # ------------------- SURGICAL CHANGE END -------------------

    # First turn / greet
    if not ctx.greeted or intent == "greet":
        ctx.greeted = True
        return T["greet"][L], ctx

    # WhatsApp / handoff
    if intent == "whatsapp_details":
        return T["whatsapp"][L], ctx
    if intent == "connect_representative":
        ctx.handoff = True
        return T["handoff"][L], ctx

    # Asking for project lists or category referenced
    if intent == "ask_projects" or detected_cat:
        # Projects request: ensure we are at the category layer
        ctx.project = None
        # If attribute requested but no project yet -> ask to pick a project within that category
        if ctx.category and ctx.attribute:
            items = _CATS.get(ctx.category, [])
            if items:
                cat_name = CAT_LABELS[ctx.category][L]
                label = ATTR_LABELS[ctx.attribute][L]
                return T["ask_project_for_attr"][L].format(
                    label=label, cat=cat_name, items=_pretty_list(items)
                ), ctx
        # Otherwise list projects for the chosen/derived category
        if ctx.category:
            items = _CATS.get(ctx.category, [])
            if items:
                cat_name = CAT_LABELS[ctx.category][L]
                return T["list_projects"][L].format(
                    cat=cat_name, items=_pretty_list(items)
                ), ctx
        # No category yet – ask for it
        return T["ask_category"][L], ctx

    # Project chosen -> answer attribute if asked, else full details
    if ctx.project:
        if ctx.attribute:
            ans = _project_answer_attr(ctx.project, ctx.attribute, L)
            if ans:
                return ans, ctx
        return _project_answer_all(ctx.project, L), ctx

    # Category chosen (but no project) -> list projects; if none, nudge attribute
    if ctx.category and not ctx.project:
        items = _CATS.get(ctx.category, [])
        if items:
            cat_name = CAT_LABELS[ctx.category][L]
            return T["list_projects"][L].format(cat=cat_name, items=_pretty_list(items)), ctx
        return T["ask_attribute"][L], ctx

    # Attribute only (no category) -> ask category
    if ctx.attribute and not ctx.category:
        return T["ask_category"][L], ctx

    # Final fallback
    return T["fallback"][L], ctx
