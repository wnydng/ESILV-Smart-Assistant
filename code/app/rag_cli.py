import json
import faiss
import numpy as np
import ollama
from pathlib import Path

# ========= Chemins absolus basés sur l'emplacement de ce fichier =========
# /ESILV-Smart-Assistant/code/app/rag_cli.py
CURRENT_DIR = Path(__file__).resolve().parent          # .../code/app
CODE_DIR = CURRENT_DIR.parent                          # .../code
VECTOR_DIR = CODE_DIR / "embeddings" / "vector_store"  # .../code/embeddings/vector_store

INDEX_PATH = VECTOR_DIR / "faiss_index.bin"
MAPPING_PATH = VECTOR_DIR / "mapping.json"

print("Chargement index et mapping...")
print(f"INDEX_PATH  = {INDEX_PATH}")
print(f"MAPPING_PATH = {MAPPING_PATH}")

# ========= Chargement FAISS + mapping =========
index = faiss.read_index(str(INDEX_PATH))

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

print("✔ Index FAISS chargé.")
print(f"Nombre de documents : {len(mapping)}\n")


# ========= FONCTIONS =========
def embed(text):
    resp = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    return np.array(resp["embedding"], dtype="float32")


def search(query, k=5):
    qvec = embed(query)
    scores, ids = index.search(np.array([qvec]), k)
    docs = []
    for score, idx in zip(scores[0], ids[0]):
        docs.append(mapping[str(idx)])
    return docs


def rag_answer(question):
    docs = search(question, k=5)

    context = ""
    for d in docs:
        context += f"- {d['content']}\n"

    prompt = f"""
Tu es l’assistant RAG de l’ESILV.
Tu dois répondre UNIQUEMENT à partir du contexte ci-dessous.
Si l'information précise n'apparaît pas, tu réponds : "Je ne sais pas.".
Ne change pas les types de diplômes : garde exactement BUT, BTS, Licence, etc.

QUESTION :
{question}

CONTEXTE :
{context}

RÉPONSE FACTUELLE ET COURTE :
"""

    result = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )
    return result["response"]


# ========= BOUCLE INTERACTIVE =========
while True:
    q = input("🧠 Question > ")
    if q.strip().lower() in ["exit", "quit"]:
        print("Au revoir !")
        break

    print("\n📚 Recherche RAG...\n")
    answer = rag_answer(q)
    print("💬 Réponse :", answer, "\n")
