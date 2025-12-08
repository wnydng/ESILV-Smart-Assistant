import json
import faiss
import numpy as np
import ollama
from pathlib import Path

# ===========================
# Chemins
# ===========================
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
VECTOR_DIR = CODE_DIR / "embeddings" / "vector_store"

INDEX_PATH = VECTOR_DIR / "faiss_index.bin"
MAPPING_PATH = VECTOR_DIR / "mapping.json"

print("Chargement index et mapping...")
print(f"INDEX_PATH  = {INDEX_PATH}")
print(f"MAPPING_PATH = {MAPPING_PATH}")

# ===========================
# Charger FAISS + mapping
# ===========================
index = faiss.read_index(str(INDEX_PATH))

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

print("✔ Index FAISS chargé.")
print(f"Nombre de documents : {len(mapping)}\n")


# ===========================
# Embedding (mxbai)
# ===========================
def embed(text):
    resp = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    return np.array(resp["embedding"], dtype="float32")


# ===========================
# Recherche FAISS
# ===========================
def search(query, k=5):
    qvec = embed(query)
    scores, ids = index.search(np.array([qvec]), k)

    docs = []
    for score, idx in zip(scores[0], ids[0]):
        docs.append(mapping[str(idx)])
    return docs


# ===========================
# RAG complet
# ===========================
def rag_answer(question):
    docs = search(question, k=5)

    context = "\n".join([d["content"] for d in docs])

    prompt = f"""
Tu es un assistant spécialisé dans l’ESILV.
Utilise UNIQUEMENT les documents suivants :

{context}

QUESTION :
{question}

Réponds de façon claire, concise et exacte.
Si la réponse n'est pas dans les documents, répond simplement : "Je ne sais pas."
"""

    result = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )

    return result["response"]


# ===========================
# Boucle interactive
# ===========================
while True:
    q = input("🧠 Question > ")
    if q.strip().lower() in ["exit", "quit"]:
        print("Au revoir !")
        break

    print("\n📚 Recherche RAG...\n")
    answer = rag_answer(q)
    print("💬 Réponse :", answer, "\n")
