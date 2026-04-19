import os
import json
import uuid
import re
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# =====================================================
# ⚙️ CONFIGURATION
# All tunable settings in one place.
# Easy to change before deploying.
# =====================================================

CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 100
TOP_K           = 4
DATA_PATH       = "data"
DB_PATH         = "db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"   # Free, fast, excellent for legal Q&A

# Chat history is saved here as a JSON file.
# This file persists across restarts and should be
# included when sharing or deploying the project.
HISTORY_FILE    = "chat_history.json"


# =====================================================
# PAGE CONFIG
# Must be the first Streamlit call.
# =====================================================

st.set_page_config(
    page_title="Mining Law Assistant",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    html, body, .stApp {
        background-color: #0c0d11;
        font-family: 'DM Sans', sans-serif;
    }

    /* Subtle noise texture overlay on main bg */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
        opacity: 0.4;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f1018;
        border-right: 1px solid #1e2035;
    }
    [data-testid="stSidebar"] > div { padding-top: 0; }

    /* ── App title block ── */
    .app-title {
        padding: 1.6rem 1rem 1rem 1rem;
        border-bottom: 1px solid #1e2035;
        margin-bottom: 0.5rem;
    }
    .app-title .icon {
        font-size: 1.6rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .app-title h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: #dde3f0;
        margin: 0 0 0.2rem 0;
        letter-spacing: 0.01em;
    }
    .app-title .subtitle {
        font-size: 0.68rem;
        color: #3a4060;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* ── Sidebar section labels ── */
    .sidebar-section {
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: #2e3450;
        text-transform: uppercase;
        padding: 1.2rem 0 0.4rem 0;
    }

    /* ── Chat bubbles ── */
    .bubble-label {
        font-size: 0.65rem;
        color: #2e3450;
        margin-bottom: 3px;
        padding-left: 6px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
    }
    .user-bubble {
        background: linear-gradient(135deg, #162844 0%, #1a3357 100%);
        border: 1px solid #1e3d6b;
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.1rem;
        margin: 0.4rem 0 0.4rem 18%;
        color: #c8ddf5;
        font-size: 0.9rem;
        line-height: 1.6;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .bot-bubble {
        background: #13151f;
        border: 1px solid #1e2238;
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.1rem;
        margin: 0.4rem 18% 0.4rem 0;
        color: #bcc5d8;
        font-size: 0.9rem;
        line-height: 1.7;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    }

    /* ── Source / citation card ── */
    .source-card {
        background: #0d0f18;
        border: 1px solid #1a1d2e;
        border-left: 2px solid #2a4a80;
        border-radius: 6px;
        padding: 0.45rem 0.85rem;
        margin: 0.15rem 18% 0.15rem 0;
        font-size: 0.72rem;
        color: #3a4560;
        font-family: 'JetBrains Mono', monospace;
    }
    .source-card strong {
        color: #5a6a90;
        font-weight: 500;
    }
    .source-card .snippet {
        color: #2a3248;
        font-style: italic;
        display: block;
        margin-top: 2px;
        font-size: 0.68rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Section badge ── */
    .section-badge {
        display: inline-block;
        background: #0f1e38;
        color: #3a6aaa;
        border: 1px solid #1a3560;
        border-radius: 3px;
        padding: 0px 5px;
        font-size: 0.65rem;
        margin-left: 6px;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    }

    /* ── Welcome screen ── */
    .welcome {
        text-align: center;
        padding: 5rem 2rem 2rem 2rem;
    }
    .welcome .welcome-icon {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    .welcome h2 {
        font-family: 'DM Serif Display', serif;
        color: #2a3050;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    .welcome p {
        font-size: 0.85rem;
        color: #252a40;
        line-height: 1.7;
    }
    .welcome .start-hint {
        margin-top: 2rem;
        font-size: 0.78rem;
        color: #1e2238;
    }
    .welcome .start-hint strong { color: #2a4a80; }

    /* ── Chat header ── */
    .chat-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #5a6a90;
        font-weight: 400;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    /* ── Credit line at bottom of sidebar ── */
    .credit-line {
        position: absolute;
        bottom: 1.2rem;
        left: 0; right: 0;
        text-align: center;
        font-size: 0.62rem;
        color: #1e2235;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }
    .credit-line span {
        color: #2a3555;
        font-weight: 500;
    }

    /* ── Info block ── */
    .info-block {
        font-size: 0.72rem;
        color: #252a3a;
        line-height: 1.9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── Streamlit button overrides ── */
    .stButton button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# CHAT HISTORY — FILE I/O
#
# All sessions stored in one JSON file structured as:
# {
#   "session-uuid": {
#     "id": "...",
#     "title": "Who appoints chief inspector?",
#     "created_at": "2024-01-01 10:00",
#     "messages": [
#       {"role": "user", "content": "...", "timestamp": "10:00"},
#       {"role": "assistant", "content": "...",
#        "citations": [...], "timestamp": "10:00"}
#     ]
#   }
# }
# =====================================================

def load_history() -> dict:
    """Load all sessions from JSON. Returns {} if file missing."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_history(history: dict):
    """Save all sessions to JSON. Called after every message."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def create_session(history: dict) -> str:
    """Create a new session, save it, return its unique ID."""
    session_id = str(uuid.uuid4())
    history[session_id] = {
        "id":         session_id,
        "title":      "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages":   []
    }
    save_history(history)
    return session_id


def delete_session(history: dict, session_id: str):
    """Remove a session from history and save."""
    if session_id in history:
        del history[session_id]
        save_history(history)


# =====================================================
# CACHED RESOURCES
# @st.cache_resource — runs only ONCE per app launch.
# Prevents reloading the model on every interaction.
# =====================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def load_db(_embeddings):
    """
    Load existing FAISS index or build from scratch.
    Underscore prefix on _embeddings tells Streamlit
    not to try hashing this argument.
    """
    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH, _embeddings,
            allow_dangerous_deserialization=True
        )

    all_documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{DATA_PATH}'.")

    for filename in pdf_files:
        loader = PDFPlumberLoader(os.path.join(DATA_PATH, filename))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        all_documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)
    db = FAISS.from_documents(chunks, _embeddings)
    db.save_local(DB_PATH)
    return db


@st.cache_resource
def load_llm():
    # Groq runs the model in the cloud — no local GPU needed.
    # The API key is stored in Streamlit secrets (never hardcoded).
    # st.secrets["GROQ_API_KEY"] reads from:
    #   - Local: .streamlit/secrets.toml
    #   - Deployed: Streamlit Cloud dashboard secrets
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.1,
        max_tokens=200,
    )


# =====================================================
# CITATION EXTRACTION
#
# After FAISS retrieves chunks, we extract:
# 1. PDF filename (source)
# 2. Page number (converted from 0-indexed to 1-indexed)
# 3. Section/Rule/Regulation number (regex scan of chunk text)
# 4. A short snippet preview of the chunk
#
# Citations are stored with each message in the JSON
# so they're visible even when revisiting old chats.
# =====================================================

def extract_citation(doc) -> dict:
    """Extract structured citation info from a retrieved chunk."""
    source = doc.metadata.get("source", "Unknown document")
    page   = doc.metadata.get("page", None)
    page_display = int(page) + 1 if isinstance(page, (int, float)) else "N/A"

    # Regex to find section/rule references in the chunk text
    # Matches: Section 5, Sec. 12, Rule 22, Regulation 4(1), Schedule II
    text  = doc.page_content
    match = re.search(
        r'\b(Section|Sec\.|Rule|Regulation|Clause|Article|Schedule)\s+(\d+[\w\(\)\.]*)',
        text,
        re.IGNORECASE
    )
    section = f"{match.group(1)} {match.group(2)}" if match else None

    return {
        "source":  source,
        "page":    page_display,
        "section": section,
        "snippet": text[:120].strip()
    }


def get_citations(docs: list) -> list:
    """Return unique citations from all retrieved chunks."""
    seen, citations = set(), []
    for doc in docs:
        cite = extract_citation(doc)
        key  = (cite["source"], cite["page"])
        if key not in seen:
            seen.add(key)
            citations.append(cite)
    return citations


# =====================================================
# RAG PIPELINE
# =====================================================

def build_prompt(context: str, question: str) -> str:
    """
    Mistral uses [INST]...[/INST] instruction format.
    We tell it explicitly to mention section numbers
    when found in the context — this enables better citations.
    """
    return f"""[INST] You are a legal assistant specializing in Indian mining law and regulations.

Use ONLY the context provided below to answer the question.
- Answer in 2-3 clear, complete sentences.
- If a specific Section or Rule number appears in the context and is relevant, mention it in your answer.
- Do not copy text verbatim from the context.
- If the answer is not in the context, say: "This information was not found in the provided documents."
- Never invent section numbers or rules not present in the context.

Context:
{context}

Question: {question} [/INST]"""


def clean_context(docs, max_chars=800):
    """Combine retrieved chunks into clean text, capped at max_chars."""
    combined, total = [], 0
    for doc in docs:
        text = " ".join(doc.page_content.split()).strip()
        if total + len(text) > max_chars:
            text = text[:max_chars - total]
        combined.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(combined)


def get_answer(question: str):
    """Full RAG pipeline: retrieve → cite → prompt → generate."""
    docs      = retriever.invoke(question)
    context   = clean_context(docs)
    prompt    = build_prompt(context, question)
    answer    = llm.invoke(prompt).content.strip()
    citations = get_citations(docs)
    return answer, citations


# =====================================================
# LOAD RESOURCES
# =====================================================

embeddings = load_embeddings()
db         = load_db(embeddings)
llm        = load_llm()

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
)


# =====================================================
# SESSION STATE INIT
# Streamlit reruns the whole script on every interaction.
# session_state persists data across those reruns.
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="app-title">
        <span class="icon">⛏️</span>
        <h1>Mining Law AI</h1>
        <div class="subtitle">Groq · Indian Mining Regulations</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        new_id = create_session(st.session_state.history)
        st.session_state.active_session_id = new_id
        st.rerun()

    st.markdown('<div class="sidebar-section">Chat History</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.markdown('<p style="color:#555; font-size:0.8rem;">No chats yet.</p>', unsafe_allow_html=True)
    else:
        # Show newest sessions first
        sorted_sessions = sorted(
            history.values(),
            key=lambda s: s.get("created_at", ""),
            reverse=True
        )

        for session in sorted_sessions:
            sid       = session["id"]
            is_active = sid == st.session_state.active_session_id
            title     = session["title"]
            short     = title[:35] + "..." if len(title) > 35 else title
            created   = session.get("created_at", "")

            col1, col2 = st.columns([5, 1])
            with col1:
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"💬 {short}", key=f"btn_{sid}",
                             use_container_width=True, type=btn_type):
                    st.session_state.active_session_id = sid
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{sid}"):
                    delete_session(st.session_state.history, sid)
                    if st.session_state.active_session_id == sid:
                        st.session_state.active_session_id = None
                    st.rerun()

            st.markdown(
                f'<p style="color:#444;font-size:0.68rem;margin:-8px 0 6px 4px;">{created}</p>',
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown('<div class="sidebar-section">ℹ️ Info</div>', unsafe_allow_html=True)
    pdf_count   = len([f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]) if os.path.exists(DATA_PATH) else 0
    total_chats = len(history)
    st.markdown(
        f'<div class="info-block">'
        f'📁 &nbsp;{pdf_count} document(s)<br>'
        f'💬 &nbsp;{total_chats} saved chat(s)<br>'
        f'🤖 &nbsp;{GROQ_MODEL}<br>'
        f'🔍 &nbsp;{TOP_K} chunks / query'
        f'</div>',
        unsafe_allow_html=True
    )

    # Subtle credit line pinned to bottom of sidebar
    st.markdown(
        '<div class="credit-line">crafted by <span>Suraj Mahato</span></div>',
        unsafe_allow_html=True
    )


# =====================================================
# MAIN CHAT AREA
# =====================================================

active_id = st.session_state.active_session_id

if active_id is None or active_id not in st.session_state.history:
    st.markdown("""
    <div class="welcome">
        <span class="welcome-icon">⛏️</span>
        <h2>Mining Law Assistant</h2>
        <p>Ask questions about the Mines Act, MMDR,<br>
        Explosive Rules, MCDR, and other Indian mining regulations.</p>
        <div class="start-hint">Click <strong>＋ New Chat</strong> in the sidebar to begin.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    session  = st.session_state.history[active_id]
    messages = session["messages"]

    st.markdown(f'<div class="chat-header">{session["title"]}</div>', unsafe_allow_html=True)
    st.caption(f"Started {session.get('created_at', '')}")
    st.divider()

    # ── Render all messages with their citations ──
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-label">You · {msg.get("timestamp","")}</div>'
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bubble-label">⛏️ Mining Law AI · {msg.get("timestamp","")}</div>'
                f'<div class="bot-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            for cite in msg.get("citations", []):
                section_html = (
                    f'<span class="section-badge">{cite["section"]}</span>'
                    if cite.get("section") else ""
                )
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>📋 {cite["source"]}</strong>{section_html} &nbsp;·&nbsp; pg. {cite["page"]}'
                    f'<span class="snippet">{cite["snippet"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Question input ──
    st.divider()
    question = st.chat_input("Ask about mining laws, regulations, rules...")

    if question:
        timestamp = datetime.now().strftime("%H:%M")

        # Record user message and save immediately
        messages.append({
            "role":      "user",
            "content":   question,
            "timestamp": timestamp
        })
        if len(messages) == 1:
            session["title"] = question[:50]
        save_history(st.session_state.history)

        # Generate answer + citations
        with st.spinner("⛏️ Searching documents and generating answer..."):
            answer, citations = get_answer(question)

        # Record assistant message with citations and save
        messages.append({
            "role":      "assistant",
            "content":   answer,
            "citations": citations,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        save_history(st.session_state.history)

        st.rerun()
