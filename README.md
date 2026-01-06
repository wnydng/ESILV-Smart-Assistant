# LLM & GenAI Project : ESILV Smart Assistant

<img width="2000" height="1050" alt="image" src="https://github.com/user-attachments/assets/922367e2-3127-4050-ab03-c7d8a2e2f90f" />

## Team 41

- Lisa  NACCACHE
- Hiba  NEJJARI
- Wendy DUONG

## Repository Overview

**ESILV Smart Assistant** is an intelligent chatbot based on a **Retrieval-Augmented Generation (RAG)** architecture, designed to provide reliable and contextualized answers to questions from students, professors, and partners of **ESILV (École Supérieure d’Ingénieurs Léonard-de-Vinci)**.

The system relies **exclusively on official ESILV sources** (institutional website pages and official PDF documents) and combines:

- Semantic vector search using **FAISS**
- Open-source embeddings generated locally with **Ollama**
- Controlled answer generation using **LLMs**
- **Specialized agents** per domain (Admissions, Programs, International, etc.)
---

## Project Context

  This project focuses on developing an Intelligent Document Processing (IDP) system capable of
extracting structured information from scanned documents (invoices, forms, certificates, IDs,
etc.) using multimodal Large Language Models (VLMs) and OCR pipelines.
The goal is to go beyond pure text extraction by combining visual understanding (layout, tables,
signatures, stamps, etc.) with semantic reasoning using Generative AI.
The system should process batches of scanned documents and produce JSON outputs containing
the extracted information in a well-defined schema.
Students are expected to experiment with open-source OCR engines (such as DeepSeek OCR or
Dots OCR) and multimodal LLMs (e.g., LLaVA, Pixtral, or Llama 3.2 Vision) deployed locally
via Ollama or integrated through Hugging Face / GCP endpoints.

## RAG Pipeline Architecture

ESILV Web Scraping
↓
Text Cleaning & Normalization
↓
Semantic Chunking
↓
Embeddings (Ollama)
↓
FAISS Vector Store
↓
Multi-source Retriever
↓
Specialized Agents
↓
LLM (Llama 3.x)
↓
Streamlit Interface


---

## 1. Targeted Scraping of the ESILV Website

### Scraped Pages

- https://www.esilv.fr/lecole/
- https://www.esilv.fr/admissions/
- https://www.esilv.fr/formations/
- https://www.esilv.fr/entreprises-debouches/
- https://www.esilv.fr/recherche/
- https://www.esilv.fr/international/

### Why Targeted Scraping?

- Avoid noise (news, blog posts, non-institutional content)
- Focus only on **stable and official pages**
- Maintain full control over data structure for RAG optimization

### Scraper Features

- Extraction of structured content:
  - H4 introduction blocks
  - H2 / H3 headings
  - Paragraphs
- Automatic detection of:
  - Content cards (`.one_third`, `.one_half`)
  - Structured lists
  - Admissions contact blocks
- Recursive scraping of **child pages**
- Generation of structured JSON files

### Structure of Each Scraped JSON

- `intro`
- `sections`
- `subrubrics`
- `child_pages`
- `full_text`

⚠️ **Scraped data is not versioned on GitHub**, in compliance with project guidelines.

---

## 2. Text Cleaning & Normalization

Before generating embeddings, the text undergoes several preprocessing steps:

- Removal of unnecessary whitespace
- Text format normalization
- Raw text extraction using **trafilatura**
- Removal of residual HTML tags
- Optimization for downstream chunking

---

## 3. Semantic Chunking

- Division into **semantic chunks**
- Chunk size optimized for RAG
- Context preservation
- Each chunk contains:
  - `content`
  - `rubric`
  - `title`
  - `url` or PDF reference

---

## 4. Embeddings & Vector Store

### Models Used (via Ollama)

- `mxbai-embed-large` → embeddings
- `llama3.2:3b` → answer generation
- `llama3.1` → testing
- `nomic-embed-text` → alternative embedding tests

### FAISS Configuration

- Index type: `IndexFlatIP`
- Similarity: cosine similarity (L2-normalized vectors)
- Two vector stores:
  - **v2** → ESILV website content
  - **v3** → ESILV official PDF documents (brochures, diplomas, etc.)

---

## 5. Project Structure

ESILV-Smart-Assistant/
│
├── data/
│ ├── scraping_esilv/ # Raw scraped JSON (not versioned)
│ └── chunks_esilv/ # Cleaned semantic chunks
│
├── code/
│ ├── embeddings/
│ │ └── vector_store_v2/
│ │ ├── faiss_index.bin
│ │ └── mapping.json
│ │
│ └── app/
│ ├── rag_agent.py # Retriever + Agents + RAG logic
│ ├── chatbot.py
│ ├── streamlit_app.py # Streamlit user interface
│
├── models/ # Managed automatically by Ollama
├── requirements.txt
└── README.md


---

## 6. Specialized Agents

Each agent is a **LLM configured with a specific role, style, and constraints**.

### Implemented Agents

- **Admissions**
  - Admission procedures, entrance exams, alternance
- **Programs / Formations**
  - Engineering cycle, majors, MSc programs
- **International**
  - Exchanges, mobility programs, partnerships

### Automatic Agent Selection

- Priority rules based on the user question
- Voting mechanism based on retrieved document rubrics
- Secure fallback strategy

---

## 7. Multi-source Retriever

The retriever queries **both vector stores simultaneously**:

- ESILV website vector store (v2)
- ESILV PDF vector store (v3)

It then:
- Merges results
- Sorts by similarity score
- Deduplicates overlapping chunks
- Builds a unified contextual prompt

---

## 8. Safety & Anti-Hallucination Mechanisms

Several safeguards are implemented:

- The model **refuses to answer** if:
  - No valid source is found
  - No URL appears in the generated answer
- Procedural hallucinations are blocked
- Standard fallback response:
  
  > *"Je n'ai pas cette information dans les documents ESILV."*

---

## 9. Streamlit Interface

### Features

- Automatic agent selection
- Display of sources and similarity scores
- Custom conversation bubbles
- Session history

### Launch

```bash
pip install -r requirements.txt
streamlit run code/app/streamlit_app.py

### Tasks repartition
à compléter

## Technologies used
à compléter
