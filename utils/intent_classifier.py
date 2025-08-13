import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

INTENT_KEYS = {
    "greet", "goodbye", "ask_projects", "whatsapp_details",
    "connect_representative", "affirm", "deny"
}

class SBERTIntentClassifier:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.intent_examples = {}
        self.labels = []
        self.embeddings = None

    def load_intents(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # keep only the intent keys
        self.intent_examples = {k: v for k, v in data.items() if k in INTENT_KEYS}
        self._raw_data = data  # expose full json for facts/config if needed

    def fit(self):
        self.labels = []
        examples = []
        for intent, phrases in self.intent_examples.items():
            for p in phrases:
                self.labels.append(intent)
                examples.append(p)
        if not examples:
            self.embeddings = None
            return
        self.embeddings = self.model.encode(examples, convert_to_tensor=False)  # numpy

    def predict(self, query: str, threshold: float = 0.6):
        if self.embeddings is None or len(self.labels) == 0:
            return "fallback", 0.0
        q_emb = self.model.encode([query], convert_to_tensor=False)  # numpy
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_label = self.labels[best_idx]
        if best_score < threshold:
            return "fallback", best_score
        return best_label, best_score

    # helpers for facts/config
    def get_facts(self):
        return self._raw_data.get("project_facts", {})

    def get_categories(self):
        return self._raw_data.get("project_categories", {})

    def get_config(self):
        return self._raw_data.get("config", {})
