# ⛏️ Mining Law AI

A RAG-based (Retrieval-Augmented Generation) chatbot for Indian mining law and regulations. Ask questions about the Mines Act, MMDR, Explosive Rules, MCDR, and other mining regulations — and get crisp, cited answers instantly.

**Built by [Suraj Mahato](https://github.com/srj17-max)**

---

## 🚀 Live Demo

> Deployed on Streamlit Community Cloud  
> [Click here to open the app](#) ← *(replace with your Streamlit URL)*

---

## 📸 Features

- 💬 **Conversational Q&A** — Ask questions in plain English
- 📄 **Source citations** — Every answer shows the PDF, page number, and section/rule reference
- 🗂️ **Persistent chat history** — All sessions saved to `chat_history.json`
- 🌙 **Dark professional UI** — Built with Streamlit and custom CSS
- ⚡ **Fast retrieval** — FAISS vector database with MMR search
- 🔒 **Local-first** — Run entirely on your own machine with Ollama

---

## 📚 Documents Covered

| Document | Description |
|---|---|
| Mines Act 1952 | Primary legislation governing mines in India |
| MMDR Act 2023 | Mines and Minerals (Development and Regulation) |
| Mines Rules 1955 | Rules under the Mines Act |
| MCDR 2017 | Mineral Conservation and Development Rules |
| Explosive Rules 2008 | Rules governing use of explosives in mines |
| MMR 1961 | Metalliferous Mines Regulations |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM (cloud) | Groq API — Mixtral 8x7B |
| LLM (local) | Ollama — Mistral 7B |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| PDF Loader | PDFPlumber |
| Framework | LangChain |

---

## 🏃 Running Locally

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/srj17-max/mining-law-ai.git
cd mining-law-ai
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**
```bash
mkdir -p .streamlit
nano .streamlit/secrets.toml
```
Add this inside:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

**5. Run the app**
```bash
streamlit run app_ui.py
```

The app will open at `http://localhost:8501`. On first run it will build the FAISS index from the PDFs (takes 2–5 minutes). Subsequent runs load instantly.

---

## 📁 Project Structure

```
mining-law-ai/
├── app_ui.py              # Main Streamlit app
├── requirements.txt       # Python dependencies
├── data/                  # PDF documents
│   ├── Mines Act 1952.pdf
│   ├── MMDR 2023.pdf
│   └── ...
├── db/                    # FAISS index (auto-generated, not in git)
├── chat_history.json      # Saved chat sessions (auto-generated)
└── .streamlit/
    └── secrets.toml       # API keys (never commit this)
```

---

## ☁️ Deploying to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, set main file to `app_ui.py`
5. Add your `GROQ_API_KEY` in the Secrets section
6. Click Deploy

---

## 📝 License

This project is for educational and research purposes. The legal documents used are publicly available Indian government publications.

---

*Crafted by Suraj Mahato*
