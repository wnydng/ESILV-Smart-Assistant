import os
import sys
import time
import io
import json
import csv
import numpy as np
import streamlit as st

# Ensure local app dir is importable
sys.path.append(os.path.dirname(__file__))

try:
    import rag_agent
except Exception as e:
    st.error(f"Impossible d'importer le module rag_agent: {e}")
    st.stop()

st.set_page_config(page_title="ESILV RAG Chatbot", layout="wide")

st.title("ESILV — Chatbot")

# Styles pour bulles conversation (utilisateur à droite, agent à gauche)
st.markdown(
    """
    <style>
    .chat-row {display:flex; align-items:flex-start; margin-bottom: 6px}
    .chat-left {flex:1; display:flex; justify-content:flex-start}
    .chat-right {flex:1; display:flex; justify-content:flex-end}
    .user {background:#DCF8C6; border-radius:12px; padding:10px; margin:6px; max-width:70%; word-wrap:break-word}
    .bot {background:#F1F0F0; border-radius:12px; padding:10px; margin:6px; max-width:70%; word-wrap:break-word}
    </style>
    """,
    unsafe_allow_html=True,
)

agent_map = {
    "Admissions": rag_agent.AGENT_ADMISSION,
    "Formations": rag_agent.AGENT_FORMATION,
    "International": rag_agent.AGENT_INTERNATIONAL,
}

with st.sidebar:
    st.header("Paramètres")
    if st.button("Réinitialiser l'historique"):
        if "history" in st.session_state:
            st.session_state.history = []

    st.markdown("**Exporter l'historique**")
    export_format = st.selectbox("Format", ["JSON", "CSV"])
    if st.button("Télécharger l'historique"):
        hist = st.session_state.get("history", [])
        if export_format == "JSON":
            payload = json.dumps(hist, ensure_ascii=False, indent=2)
            st.download_button("Download JSON", data=payload, file_name="history.json", mime="application/json")
        else:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["question", "answer", "agent", "found", "retrieval_time", "gen_time", "sources"])
            for item in hist:
                sources_txt = " | ".join([f"{s.get('title','')}<{s.get('url','')}>" for s in item.get("sources", [])])
                writer.writerow([
                    item.get("question",""),
                    item.get("answer",""),
                    item.get("agent",""),
                    item.get("found", False),
                    item.get("retrieval_time", ""),
                    item.get("gen_time", ""),
                    sources_txt,
                ])
            st.download_button("Download CSV", data=buf.getvalue(), file_name="history.csv", mime="text/csv")

# Paramètres fixes (non modifiables par l'utilisateur)
top_k = 10
truncate_chars = 200

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_area("Question", height=130)

if st.button("Envoyer") and question.strip():
    # Indiquer que l'agent tourne
    with st.spinner("Agent en cours..."):
        # 1) Détection d'agent (utilise FAISS en interne)
        agent_choice = rag_agent.detect_agent(question)
        agent_prompt = agent_map[agent_choice]

        # 2) Retrieval (mesurer le temps)
        retrieval_start = time.time()
        sources = []
        try:
            q_vec = rag_agent.embed(question)
            q_vec = np.expand_dims(q_vec, axis=0)
            D, I = rag_agent.index.search(q_vec, top_k)
            for score, idx in zip(D[0], I[0]):
                key = str(int(idx))
                if key in rag_agent.mapping:
                    doc = rag_agent.mapping[key]
                    snippet = doc.get("content", "")[: int(truncate_chars)].replace("\n", " ")
                    sources.append({"score": float(score), "title": doc.get("title"), "url": doc.get("url"), "snippet": snippet, "rubric": doc.get("rubric","")})
        except Exception as e:
            st.warning(f"Impossible de récupérer les sources (FAISS/Ollama) : {e}")
        retrieval_time = time.time() - retrieval_start

        # 3) Génération (mesurer le temps)
        gen_start = time.time()
        try:
            answer = rag_agent.ask_agent(question, agent_prompt, top_k=top_k)
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'agent : {e}")
            answer = None
        gen_time = time.time() - gen_start

    # Déterminer si l'agent a trouvé la réponse pertinente
    found = False
    if answer:
        a_lower = answer.lower()
        if ("je ne sais pas" in a_lower) or ("je n'ai pas" in a_lower) or ("i don't know" in a_lower):
            found = False
        else:
            # considérer trouvé si on a des sources récupérées
            found = len(sources) > 0

    if answer is not None:
        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "agent": agent_choice,
            "sources": sources,
            "retrieval_time": round(retrieval_time, 3),
            "gen_time": round(gen_time, 3),
            "found": found,
        })

for i, turn in enumerate(reversed(st.session_state.history)):
    # Question (utilisateur) — alignée à droite
    st.markdown(
        '<div class="chat-row"><div class="chat-left"></div><div class="chat-right"><div class="user">'
        + turn['question'].replace('\n', '<br>') + '</div></div></div>',
        unsafe_allow_html=True,
    )

    # Réponse (bot) — alignée à gauche
    st.markdown(
        '<div class="chat-row"><div class="chat-left"><div class="bot">'
        + turn['answer'].replace('\n', '<br>') + '</div></div><div class="chat-right"></div></div>',
        unsafe_allow_html=True,
    )

    # Afficher temps uniquement si trouvé
    if turn.get('found'):
        st.caption(f"Recherche: {turn.get('retrieval_time','')}s • Génération: {turn.get('gen_time','')}s")

    # Feedback
    col1, col2, col3 = st.columns([1,1,4])
    useful_key = f"useful_{i}"
    notuseful_key = f"notuseful_{i}"
    comment_key = f"comment_{i}"
    if col1.button("Utile", key=useful_key):
        st.session_state.setdefault('feedback', []).append({"index": i, "useful": True})
    if col2.button("Pas utile", key=notuseful_key):
        st.session_state.setdefault('feedback', []).append({"index": i, "useful": False})
    comment = col3.text_input("Commentaire (optionnel)", key=comment_key)
    if comment:
        st.session_state.setdefault('feedback_comments', []).append({"index": i, "comment": comment})

    # Sources (discrètes)
    if turn.get('sources'):
        with st.expander("Sources"):
            for s in turn['sources']:
                st.markdown(f"- **{s.get('title','')}** — <{s.get('url','')}>")
                # Afficher score discrètement
                st.caption(f"Score: {s['score']:.3f} — {s.get('snippet')}")

    st.markdown("---")
