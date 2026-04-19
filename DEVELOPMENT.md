# Mining Law AI — Development Document

**Project:** Mining Law AI Chatbot  
**Developer:** Suraj Mahato  
**Version:** 1.0  
**Date:** April 2026  
**Stack:** Python · LangChain · FAISS · Groq API · Streamlit

---

## 1. Project Overview

Mining Law AI is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indian mining law. It retrieves relevant passages from a curated set of legal PDFs and uses a large language model to generate concise, cited answers.

The system is designed to be deployable on Streamlit Community Cloud with zero infrastructure cost, using Groq's free API tier for LLM inference.

---

## 2. Architecture

### 2.1 High-Level Flow

```
User Question
     ↓
Embed question (all-MiniLM-L6-v2)
     ↓
FAISS MMR Search → Top 4 relevant chunks
     ↓
Build prompt with context + question
     ↓
Groq API (Mixtral-8x7B) → Generate answer
     ↓
Extract citations (source, page, section)
     ↓
Display answer + citations in Streamlit UI
     ↓
Save to chat_history.json
```

### 2.2 Component Breakdown

| Component | Technology | Purpose |
|---|---|---|
| PDF Loading | PDFPlumber | Extracts text from legal PDFs with better layout handling than pypdf |
| Text Splitting | RecursiveCharacterTextSplitter | Splits on paragraphs → sentences → words to avoid mid-sentence cuts |
| Embeddings | all-MiniLM-L6-v2 | Converts text to 384-dimensional vectors for similarity search |
| Vector Store | FAISS | Facebook AI Similarity Search — fast local index, no external DB needed |
| Retrieval | MMR (Maximal Marginal Relevance) | Retrieves relevant AND diverse chunks — prevents 4 identical results |
| LLM (cloud) | Groq — Mixtral-8x7B | Cloud inference, fast, free tier, instruction-tuned |
| LLM (local) | Ollama — Mistral 7B | Local inference for development on Apple Silicon |
| Frontend | Streamlit | Python-native web framework, deployable with one click |
| History | JSON file | Persistent chat sessions, portable across deployments |

---

## 3. Configuration Parameters

All parameters are defined at the top of `app_ui.py` for easy tuning:

| Parameter | Current Value | Effect |
|---|---|---|
| `CHUNK_SIZE` | 700 chars | Size of each text chunk. Too small = loses context. Too large = noisy retrieval. |
| `CHUNK_OVERLAP` | 100 chars | Overlap between chunks. Prevents answers cut off at boundaries. |
| `TOP_K` | 4 | Number of chunks retrieved per query. More = richer context, more noise. |
| `GROQ_MODEL` | mixtral-8x7b-32768 | LLM model used for answer generation. |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Embedding model. Faster alternatives exist but lower accuracy. |

---

## 4. Key Design Decisions

### 4.1 Why FAISS over ChromaDB or Pinecone?
FAISS runs entirely locally with no external service, no API calls, and no cost. The index is saved as a folder and can be committed to a repo or bundled with deployment. For a document set under 1000 pages, FAISS performance is more than sufficient.

### 4.2 Why MMR retrieval over plain similarity search?
Plain similarity search can return 4 chunks from the same paragraph if the query matches that paragraph strongly. MMR balances relevance with diversity — each retrieved chunk must be relevant but also different from already-selected chunks. This gives the LLM broader context.

### 4.3 Why Groq over OpenAI?
Groq offers a free tier with generous rate limits. Mixtral-8x7B on Groq is faster than GPT-3.5 and comparable in quality for legal Q&A. No credit card required for the free tier.

### 4.4 Why RecursiveCharacterTextSplitter?
Legal documents have deeply nested structure: parts → chapters → sections → sub-sections → clauses. RecursiveCharacterTextSplitter attempts splits in order: paragraph breaks → line breaks → sentence ends → word boundaries. This preserves legal sentence structure far better than simple character splitting.

### 4.5 Why PDFPlumber over PyPDF?
Legal PDFs from Indian government sources often have complex column layouts, headers, and footers. PDFPlumber handles these better and extracts cleaner text with more accurate page metadata.

---

## 5. Prompt Engineering

The LLM prompt uses Mistral/Mixtral's native `[INST]...[/INST]` instruction format:

```
[INST] You are a legal assistant specializing in Indian mining law.

Use ONLY the context provided below to answer the question.
- Answer in 2-3 clear, complete sentences.
- If a specific Section or Rule number appears in the context, mention it.
- Do not copy text verbatim.
- If not found, say: "This information was not found in the provided documents."
- Never invent section numbers or rules.

Context: {retrieved chunks}
Question: {user question} [/INST]
```

Key constraints enforced in the prompt:
- Answer length capped at 2–3 sentences (prevents rambling)
- Section number citation required when present (improves answer quality)
- Hallucination prevention ("never invent" instruction)
- Graceful fallback when answer not found

---

## 6. Citation Extraction

After retrieval, each chunk is scanned with a regex to extract legal references:

```python
re.search(
    r'\b(Section|Sec\.|Rule|Regulation|Clause|Article|Schedule)\s+(\d+[\w\(\)\.]*)',
    text, re.IGNORECASE
)
```

This captures references like:
- `Section 5`
- `Rule 22(1)`
- `Regulation 4`
- `Schedule II`
- `Clause 3(a)`

Citations are stored with each message in `chat_history.json` so they persist when revisiting old chats.

---

## 7. Chat History Schema

```json
{
  "session-uuid": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Who appoints the Chief Inspector?",
    "created_at": "2026-04-19 10:30",
    "messages": [
      {
        "role": "user",
        "content": "Who appoints the Chief Inspector?",
        "timestamp": "10:30"
      },
      {
        "role": "assistant",
        "content": "The Central Government appoints the Chief Inspector under Section 5...",
        "citations": [
          {
            "source": "Mines Act 1952.pdf",
            "page": 5,
            "section": "Section 5",
            "snippet": "The Central Government may appoint..."
          }
        ],
        "timestamp": "10:30"
      }
    ]
  }
}
```

---

## 8. Deployment

### Local (Development)
- LLM: Ollama + Mistral 7B running on `localhost:11434`
- Secrets: `.streamlit/secrets.toml` (never committed)
- Index: `db/` folder (auto-built on first run)

### Cloud (Production)
- LLM: Groq API (Mixtral-8x7B)
- Secrets: Streamlit Cloud secrets dashboard
- Index: Rebuilt on first deploy from `data/` PDFs
- Platform: Streamlit Community Cloud (free tier)

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| FAISS index rebuilds on cloud cold start | ~3–5 min delay on first deploy | Acceptable for free tier |
| Groq free tier rate limits | ~30 requests/minute | Sufficient for personal/small team use |
| Regex citation extraction misses complex references | Some sections not tagged | Improvement area — see Section 10 |
| Chat history not user-isolated on cloud | All users share one history file | Acceptable for single-user or small team |
| PDFs must be text-based | Scanned PDFs return no text | Pre-process scanned PDFs with OCR |

---

## 10. Improvement Roadmap

### 10.1 Data Improvements
- Add more mining regulation PDFs (state-specific rules, circulars, notifications)
- Pre-process PDFs with OCR (Tesseract) for scanned documents
- Add metadata tagging per document (Act name, year, ministry)

### 10.2 Model Improvements
- Switch embedding model to `BAAI/bge-large-en` for better legal text understanding
- Fine-tune a small model on mining law Q&A pairs
- Experiment with larger `CHUNK_SIZE` (900–1000) for longer legal provisions

### 10.3 Retrieval Improvements
- Add a reranker (CrossEncoder) to rerank retrieved chunks before sending to LLM
- Implement hybrid search (BM25 keyword + FAISS semantic)
- Add query expansion for legal synonyms

### 10.4 Application Improvements
- User authentication for isolated chat histories
- Export chat as PDF
- Answer confidence score display
- Multi-language support (Hindi)

---

## 11. File Structure

```
mining-law-ai/
├── app_ui.py              # Main application — all logic and UI
├── requirements.txt       # Python package dependencies
├── README.md              # Public-facing project description
├── DEVELOPMENT.md         # This document
├── data/                  # Source PDF documents
│   ├── Mines Act 1952.pdf
│   ├── MMDR 2023.pdf
│   ├── Mines Rules 1955.pdf
│   ├── MCDR 2017.pdf
│   ├── Explosive Rules 2008.pdf
│   └── MMR 1961.pdf
├── db/                    # FAISS index (auto-generated, gitignored)
├── chat_history.json      # Session storage (auto-generated, gitignored)
└── .streamlit/
    └── secrets.toml       # API keys (never committed)
```

---

## 12. Dependencies

```
streamlit                  # Web UI framework
langchain-community        # PDF loaders, FAISS integration
langchain-text-splitters   # RecursiveCharacterTextSplitter
langchain-huggingface      # Embedding model wrapper
langchain-groq             # Groq API LLM wrapper
faiss-cpu                  # Vector similarity search
pdfplumber                 # PDF text extraction
sentence-transformers      # all-MiniLM-L6-v2 embedding model
```

---

*Document maintained by Suraj Mahato*  
*Last updated: April 2026*
