import os
import sys
import time
import io
import json
import base64
import re
import datetime
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
# Thème adapté ESILV : accents en #ce1052 ; on conserve la couleur des bulles définies plus bas
st.markdown(
    """
    <style>
    :root { --esilv-primary: #ce1052; --esilv-dark: #8a003f; }

    /* Page global */
    body {
        background: #ffffff;
        color: #111111;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        border-left: 6px solid var(--esilv-primary);
    }
    section[data-testid="stSidebar"] h2 {
        color: var(--esilv-primary);
    }

    /* Headings and accents */
    h1, h2, h3 { color: var(--esilv-primary); }
    a { color: var(--esilv-primary); }

    /* Buttons (send / download) */
    .stButton>button, .stDownloadButton>button {
        background-color: var(--esilv-primary) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        border: none !important;
    }

    /* Chat bubbles layout */
    .chat-row {display:flex; align-items:flex-start; margin-bottom: 6px}
    .chat-left {flex:1; display:flex; justify-content:flex-start}
    .chat-right {flex:1; display:flex; justify-content:flex-end}
    /* conserver les couleurs des bulles (ne pas modifier) */
    .user {background:#ffffff; color:#000000; border-radius:12px; padding:10px; margin:6px; max-width:70%; word-wrap:break-word}
    .bot {background:#ce1052; color:#ffffff; border-radius:12px; padding:10px; margin:6px; max-width:70%; word-wrap:break-word}

    /* Expander header accent */
    .streamlit-expanderHeader { color: var(--esilv-primary); }
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
    # Logo (base64 PNG) affiché en haut à gauche, au-dessus du panneau Paramètres
    try:
        img_b64 = (
            "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"
        )
        img_bytes = base64.b64decode(img_b64)
        st.image(img_bytes, width=120)
    except Exception:
        pass
    st.header("Paramètres")

    # --- Contact form (required fields only) ---
    st.subheader("S'inscrire / Être recontacté")
    CONTACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "contacts")
    os.makedirs(CONTACTS_DIR, exist_ok=True)
    CONTACTS_CSV = os.path.join(CONTACTS_DIR, "contacts.csv")

    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    with st.form(key='contact_form'):
        first_name = st.text_input('Prénom', key='cf_first')
        last_name = st.text_input('Nom', key='cf_last')
        email = st.text_input('Email', key='cf_email')
        consent = st.checkbox("J'accepte d'être contacté(e) par l'ESILV pour les finalités indiquées.", key='cf_consent')
        submit_contact = st.form_submit_button('Envoyer ma demande')

    if submit_contact:
        errors = []
        if not first_name or not first_name.strip():
            errors.append("Veuillez indiquer votre prénom.")
        if not last_name or not last_name.strip():
            errors.append("Veuillez indiquer votre nom.")
        if not email or not email_re.match(email.strip()):
            errors.append("Veuillez indiquer une adresse e-mail valide.")
        if not consent:
            errors.append("Vous devez accepter d'être contacté(e) pour que nous enregistrions vos coordonnées.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            row = {
                'first_name': first_name.strip(),
                'last_name': last_name.strip(),
                'email': email.strip(),
                'consent': True,
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'source': 'chatbot_sidebar',
            }

            write_header = not os.path.exists(CONTACTS_CSV)
            try:
                with open(CONTACTS_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                st.success('Merci — vos coordonnées ont bien été enregistrées. Nous vous contacterons bientôt.')

                # Optionnel : ajouter une confirmation dans l'historique du chat
                conf_msg = "Vos coordonnées ont été enregistrées pour suivi par l'ESILV."
                st.session_state.history.append({
                    'question': f"(Inscription) {first_name} {last_name}",
                    'answer': conf_msg,
                    'agent': 'Admissions',
                    'sources': [],
                    'retrieval_time': 0,
                    'gen_time': 0,
                    'found': False,
                })
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement des coordonnées : {e}")
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
