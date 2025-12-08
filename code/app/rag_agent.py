import json
import numpy as np
import faiss
import ollama
import os

# ----------------------------------------------------
# Charger index FAISS + mapping
# ----------------------------------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_PATH, "../embeddings/vector_store")

INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
MAP_PATH   = os.path.join(VECTOR_DIR, "mapping.json")

print("Chargement index et mapping...")

index = faiss.read_index(INDEX_PATH)

with open(MAP_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

print("✔ Index FAISS chargé.")
print(f"Nombre de documents : {len(mapping)}")


# ----------------------------------------------------
# EMBEDDING MODEL (Ollama)
# ----------------------------------------------------

def embed(text):
    resp = ollama.embeddings(
        model="mxbai-embed-large",
        prompt=text
    )
    return np.array(resp["embedding"], dtype="float32")


# ----------------------------------------------------
# AGENT: prompt system
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
# DÉTECTION AUTOMATIQUE D'AGENT
# ----------------------------------------------------

AGENT_KEYWORDS = {
    "Admissions": ["admissions"],
    "Formations": ["formations"],
    "International": ["international"],
}

def detect_agent(question: str, top_k: int = 5) -> str:
    """
    Détecte l'agent approprié basé sur les rubrics des documents retrouvés.
    Récupère les top_k documents via FAISS et compte les rubrics.
    """
    try:
        # Embedding de la question
        q_vec = embed(question)
        q_vec_array = np.expand_dims(q_vec, axis=0)
        
        # Recherche FAISS
        D, I = index.search(q_vec_array, top_k)
        
        # Compter les rubrics
        rubric_counts = {}
        for idx in I[0]:
            doc = mapping.get(str(idx), {})
            rubric = doc.get("rubric", "").lower()
            
            # Mapper rubric vers agent
            if "admission" in rubric or "concour" in rubric:
                agent = "Admissions"
            elif "formation" in rubric or "programme" in rubric or "cursus" in rubric:
                agent = "Formations"
            elif "international" in rubric or "échange" in rubric or "mobilité" in rubric:
                agent = "International"
            else:
                agent = "Admissions"  # Par défaut
            
            rubric_counts[agent] = rubric_counts.get(agent, 0) + 1
        
        # Retourner l'agent avec le plus de hits
        if rubric_counts:
            best_agent = max(rubric_counts, key=rubric_counts.get)
            return best_agent
        
        return "Admissions"  # Fallback
    except Exception as e:
        print(f"⚠️ Erreur lors de la détection d'agent : {e}")
        return "Admissions"  # Fallback en cas d'erreur


# ----------------------------------------------------
# GENERATION LLM VIA OLLAMA (llama3.2:3b-instruct)
# ----------------------------------------------------

def generate_answer(context, question, agent_prompt):
    prompt = f"""
{agent_prompt}

CONTEXTE (extraits pertinents) :
{context}

QUESTION :
{question}

Réponds uniquement en utilisant ces informations.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


# ----------------------------------------------------
# PIPELINE RAG (retrieval + LLM)
# ----------------------------------------------------

def ask_agent(question, agent_prompt, top_k=3):
    # 1. Embedding de la question
    q_vec = embed(question)
    q_vec = np.expand_dims(q_vec, axis=0)

    # 2. Recherche FAISS
    D, I = index.search(q_vec, top_k)

    context = ""
    for idx in I[0]:
        doc = mapping[str(idx)]
        snippet = doc["content"][:300].replace("\n", " ")
        context += f"- {snippet}... (source: {doc['url']})\n"

    # 3. Génération finale via agent
    return generate_answer(context, question, agent_prompt)
