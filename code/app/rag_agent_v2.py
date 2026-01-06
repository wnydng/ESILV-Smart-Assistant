# rag_agent.py
# Multi-store RAG (v2 site + v3 pdf) with Ollama embeddings + FAISS
# - Keeps backward compatibility: exposes `index` and `mapping` (points to v2)
# - Provides: rag_agent (MultiStoreRAG), detect_agent(), ask_agent()

import json
import os
import re
import numpy as np
import faiss
import ollama

# ----------------------------------------------------
# EMBEDDING MODEL (Ollama)
# ----------------------------------------------------

DEBUG_EMBED = False

def embed(text: str) -> np.ndarray:
    resp = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    v = np.array(resp["embedding"], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    if DEBUG_EMBED:
        print("embed() called, norm=", float(np.linalg.norm(v)))
    return v


# ----------------------------------------------------
# LOAD VECTOR STORE
# ----------------------------------------------------

def load_store(vector_dir: str):
    index_path = os.path.join(vector_dir, "faiss_index.bin")
    map_path   = os.path.join(vector_dir, "mapping.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Missing mapping.json: {map_path}")

    index = faiss.read_index(index_path)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    print(f"Loaded store: {vector_dir}")
    print("  index type:", type(index))
    try:
        print("  metric_type:", index.metric_type)
    except Exception:
        pass
    print("  index.d:", index.d)
    print("  mapping size:", len(mapping))

    return index, mapping


# ----------------------------------------------------
# MULTI-STORE SEARCH
# ----------------------------------------------------

class MultiStoreRAG:
    def __init__(self, stores):
        """
        stores: list of dicts:
          [
            {"name": "v2_site", "index": index_v2, "mapping": mapping_v2},
            {"name": "v3_pdf",  "index": index_v3, "mapping": mapping_v3},
          ]
        """
        self.stores = stores

    def search(self, question: str, top_k_per_store: int = 6, top_k_total: int = 10):
        q = embed(question).reshape(1, -1)

        hits = []
        for s in self.stores:
            D, I = s["index"].search(q, top_k_per_store)

            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                key = str(int(idx))
                doc = s["mapping"].get(key, {})
                if not doc:
                    continue
                hits.append({
                    "score": float(score),
                    "idx": int(idx),
                    "store": s["name"],
                    "doc": doc,
                })

        # sort by score desc
        hits.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate: (url + first 80 chars of content)
        seen = set()
        dedup = []
        for h in hits:
            url = (h["doc"].get("url") or "").strip()
            content = (h["doc"].get("content") or "").strip()
            sig = (url, content[:80])
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(h)
            if len(dedup) >= top_k_total:
                break

        return dedup


# ----------------------------------------------------
# BUILD RAG AGENT (v2 + v3)
# ----------------------------------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

V2_DIR = os.path.join(BASE_PATH, "../embeddings/vector_store_v2")
V3_DIR = os.path.join(BASE_PATH, "../embeddings/vector_store_v3")  # PDFs

index_v2, mapping_v2 = load_store(V2_DIR)
index_v3, mapping_v3 = load_store(V3_DIR)

rag_agent = MultiStoreRAG([
    {"name": "v2_site", "index": index_v2, "mapping": mapping_v2},
    {"name": "v3_pdf",  "index": index_v3, "mapping": mapping_v3},
])

# Backward compatibility for old app code:
# Some code still expects `rag_agent.index` and `rag_agent.mapping`.
# We expose them here (pointing to v2).
index = index_v2
mapping = mapping_v2


# ----------------------------------------------------
# SAFETY / GUARDRAILS
# ----------------------------------------------------

PROCEDURE_HINTS = [
    "candidater", "candidature", "dossier", "étude de dossier", "entretien",
    "admission", "inscription", "modalités", "procédure"
]

def context_has_procedure(context: str) -> bool:
    c = context.lower()
    return any(h in c for h in PROCEDURE_HINTS)

def safe_answer(answer: str, context: str) -> str:
    if not answer:
        return "Je n'ai pas cette information dans les documents ESILV."

    # extract urls (robust)
    urls = [tok.strip("()[],:") for tok in context.split() if "http" in tok or tok.startswith("pdf://")]

    # at least one URL must be present in answer (if context has urls)
    if urls and not any(u in answer for u in urls):
        return "Je n'ai pas cette information dans les documents ESILV."

    return answer


# ----------------------------------------------------
# AGENTS PROMPTS
# ----------------------------------------------------

AGENT_ADMISSION = """
Tu es l'expert officiel des admissions de l'ESILV.
Tu réponds uniquement à partir du contexte fourni.
Si l'information n'est pas dans les documents, tu dis :
"Je n'ai pas cette information dans les documents ESILV."

Style :
- clair
- court
- structuré
- renvoie toujours la source
"""

AGENT_FORMATION = """
Tu es un conseiller pédagogique de l'ESILV.
Tu réponds uniquement avec le contexte fourni.
Si la formation n’existe pas → tu le dis.
"""

AGENT_INTERNATIONAL = """
Tu es l'expert mobilité internationale de l'ESILV.
Tu n'inventes rien.
"""


# ----------------------------------------------------
# AGENT DETECTION (multi-store)
# ----------------------------------------------------

def detect_agent(question: str, top_k_total: int = 8) -> str:
    q_lower = question.lower()

    # Priority rules from the question itself
    if "alternance" in q_lower or "apprentissage" in q_lower:
        return "Admissions"
    if "international" in q_lower or "mobilité" in q_lower or "échange" in q_lower:
        return "International"

    hits = rag_agent.search(question, top_k_per_store=4, top_k_total=top_k_total)

    rubric_counts = {}
    for h in hits:
        rubric = (h["doc"].get("rubric") or "").lower()

        if "admission" in rubric or "concour" in rubric:
            agent = "Admissions"
        elif "formation" in rubric or "programme" in rubric or "cursus" in rubric:
            agent = "Formations"
        elif "international" in rubric or "échange" in rubric or "mobilité" in rubric:
            agent = "International"
        else:
            agent = "Admissions"

        rubric_counts[agent] = rubric_counts.get(agent, 0) + 1

    if rubric_counts:
        return max(rubric_counts, key=rubric_counts.get)

    return "Admissions"


# ----------------------------------------------------
# LLM GENERATION (Ollama)
# ----------------------------------------------------

def generate_answer(context: str, question: str, agent_prompt: str) -> str:
    messages = [
        {"role": "system", "content": agent_prompt},
        {
            "role": "user",
            "content": (
                f"CONTEXTE:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "Regles:\n"
                "- Reponds uniquement avec le contexte.\n"
                "- Si le contexte ne contient pas la reponse, dis: "
                "\"Je n'ai pas cette information dans les documents ESILV.\"\n"
                "- N'invente rien.\n"
                "- Si tu reponds, cite au moins une URL presente dans le CONTEXTE (https://... ou pdf://...).\n"
            ),
        },
    ]
    resp = ollama.chat(model="llama3.2:3b", messages=messages)
    return resp["message"]["content"]


# ----------------------------------------------------
# RAG PIPELINE
# ----------------------------------------------------

def ask_agent(question: str, agent_prompt: str, top_k: int = 10):
    hits = rag_agent.search(
        question,
        top_k_per_store=8,
        top_k_total=top_k
    )

    print("---- RETRIEVAL DEBUG (MULTI-STORE) ----")
    for h in hits:
        doc = h["doc"]
        print(
            f'{h["score"]:.4f}',
            "| store:", h["store"],
            "|", doc.get("rubric"),
            "|", doc.get("title"),
            "|", doc.get("url"),
        )
    print("--------------------------------------")

    # Build context
    context_lines = []
    for h in hits:
        doc = h["doc"]
        snippet = (doc.get("content") or "")
        snippet = snippet[:1100].replace("\n", " ")
        url = doc.get("url", "")
        store = h["store"]
        context_lines.append(f"- [{store}] {snippet} (source: {url})")

    context = "\n".join(context_lines)

    answer = generate_answer(context, question, agent_prompt)

    # Optional: block "invented procedures"
    if (("1." in answer or "2." in answer or "étape" in answer.lower())
            and not context_has_procedure(context)):
        return "Je n'ai pas cette information dans les documents ESILV."

    return safe_answer(answer, context)


# ----------------------------------------------------
# Quick manual test
# ----------------------------------------------------

if __name__ == "__main__":
    qs = [
        "Comment candidater en alternance ?",
        "Quelles majeures existe pour le diplôme ingénieur ESILV ?",
        "Qu'est-ce que le Devinci Research Center ?",
    ]

    for q in qs:
        agent = detect_agent(q)
        if agent == "Admissions":
            prompt = AGENT_ADMISSION
        elif agent == "Formations":
            prompt = AGENT_FORMATION
        else:
            prompt = AGENT_INTERNATIONAL

        print("\nQUESTION:", q)
        print("AGENT:", agent)
        print(ask_agent(q, prompt, top_k=10))
