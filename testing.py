from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer("models/sbert/paraphrase-MiniLM-L6-v2")
    print("✅ SBERT model loaded successfully!")

    emb = model.encode("This is a test.")
    print("✅ Embedding shape:", emb.shape)
except Exception as e:
    print("❌ Still broken:", e.__class__.__name__, "-", str(e))
