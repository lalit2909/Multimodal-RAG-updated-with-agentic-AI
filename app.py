"""
Streamlit app for the Enhanced Multimodal RAG with Agentic Architecture.

Layout:
- Sidebar: Gemini API key, file uploader, KB stats, source list, clear button.
- Main:    Chat interface with source-attribution badges and an expandable
           "Agent reasoning trace" panel showing each step the agent took.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Optional

import streamlit as st

# Make `core` importable when running from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import RAGAgent, AgentResult  # noqa: E402
from core.document_processor import process_file  # noqa: E402
from core.gemini_client import GeminiClient  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_app")

# ---------- Page config ----------

st.set_page_config(
    page_title="Enhanced Multimodal RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session state ----------

def _init_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(persist_dir="data/chroma_db")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {role, content, result}
    if "gemini" not in st.session_state:
        st.session_state.gemini = None  # type: Optional[GeminiClient]
    if "agent" not in st.session_state:
        st.session_state.agent = None  # type: Optional[RAGAgent]


_init_state()


def _get_gemini(api_key: str) -> GeminiClient:
    """Instantiate (or refresh) the Gemini client after key entry."""
    if st.session_state.gemini is None or st.session_state.get("_api_key") != api_key:
        st.session_state.gemini = GeminiClient(api_key=api_key)
        st.session_state.agent = RAGAgent(
            gemini=st.session_state.gemini,
            vector_store=st.session_state.vector_store,
        )
        st.session_state._api_key = api_key
    return st.session_state.gemini


# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("## 🔑 Gemini API Key")
    api_key = st.text_input(
        "Paste your Gemini API key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    if api_key:
        try:
            _get_gemini(api_key)
            st.success("Gemini client ready.")
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}")
            st.session_state.gemini = None
            st.session_state.agent = None

    st.divider()
    st.markdown("## 📚 Knowledge Base")

    stats = st.session_state.vector_store.stats()
    st.metric("Total chunks stored", stats["total_chunks"])
    sources = st.session_state.vector_store.list_sources()
    st.caption(f"Sources: {len(sources)}")
    if sources:
        with st.expander("See sources", expanded=False):
            for s in sources:
                st.markdown(f"- {s}")

    st.markdown("### Upload files")
    uploaded = st.file_uploader(
        "PDF, image, audio, or video",
        type=["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp",
              "wav", "mp3", "aac", "flac", "m4a", "ogg",
              "mp4", "mpeg", "mov", "avi", "webm",
              "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.session_state.gemini is None:
        st.warning("Enter your Gemini API key first.")

    if uploaded and st.session_state.gemini is not None:
        if st.button("➕ Ingest into Knowledge Base", type="primary", use_container_width=True):
            progress = st.progress(0.0, text="Starting ingestion...")
            total_added = 0
            for i, f in enumerate(uploaded):
                progress.progress((i / len(uploaded)), text=f"Processing {f.name}...")
                # Save upload to a temp file so processors can open by path.
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getbuffer())
                    tmp_path = tmp.name
                try:
                    docs = process_file(tmp_path, st.session_state.gemini)
                    added = st.session_state.vector_store.add_documents(docs, st.session_state.gemini)
                    total_added += added
                    st.toast(f"{f.name}: {added} chunks added", icon="✅")
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
                    logger.exception("Ingestion failed for %s", f.name)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            progress.progress(1.0, text="Done!")
            st.success(f"Ingested {total_added} new chunks.")
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.chat_history = []
        st.toast("Knowledge base cleared.", icon="🧹")
        st.rerun()

    st.divider()
    st.caption("Built with Gemini + ChromaDB + DuckDuckGo")


# ---------- Main panel ----------

st.title("🧠 Enhanced Multimodal RAG Agent")
st.markdown(
    "Ask a question. The agent first searches the **local knowledge base**, "
    "evaluates whether it has enough context, and only falls back to a "
    "**web search** if needed. The answer is clearly labeled with its source."
)

# Source badge legend.
badge_cols = st.columns(3)
with badge_cols[0]:
    st.markdown("🟢 **Local Knowledge Base**")
with badge_cols[1]:
    st.markdown("🔵 **Internet Sources**")
with badge_cols[2]:
    st.markdown("🟣 **Both**")

st.divider()

# ---------- Chat history rendering ----------

for entry in st.session_state.chat_history:
    role = entry["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(entry["content"])
        else:
            result: AgentResult = entry["result"]
            source = result.source
            if source == "Local Knowledge Base":
                color, emoji = "green", "🟢"
            elif source == "Internet Sources":
                color, emoji = "blue", "🔵"
            else:
                color, emoji = "purple", "🟣"
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{emoji} Source: {source}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(result.answer)

            # Agent reasoning trace.
            with st.expander("🤖 Agent reasoning trace", expanded=False):
                for step in result.trace:
                    st.markdown(f"**{step.step}** — {step.detail}")

            # Local evidence.
            if result.local_chunks:
                with st.expander(f"📚 Local evidence ({len(result.local_chunks)} chunks)", expanded=False):
                    for i, c in enumerate(result.local_chunks, 1):
                        st.markdown(
                            f"**Chunk {i}** — `{c['source']}` "
                            f"({c['modality']}, sim={c['similarity']:.2f})"
                        )
                        st.caption(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))

            # Web evidence.
            if result.web_chunks:
                with st.expander(f"🌐 Web evidence ({len(result.web_chunks)} results)", expanded=False):
                    for i, r in enumerate(result.web_chunks, 1):
                        st.markdown(f"**Result {i}** — [{r.get('title', 'no title')}]({r.get('url', '#')})")
                        snippet = r.get("snippet", "") or r.get("full_text", "")
                        st.caption(snippet[:400] + ("..." if len(snippet) > 400 else ""))


# ---------- Chat input ----------

prompt = st.chat_input("Ask a question about your knowledge base...")

if prompt:
    if st.session_state.gemini is None or st.session_state.agent is None:
        st.error("Please enter your Gemini API key in the sidebar first.")
    else:
        # Save user message.
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the agent.
        with st.chat_message("assistant"):
            with st.spinner("Agent thinking..."):
                try:
                    result = st.session_state.agent.run(prompt)
                except Exception as e:
                    logger.exception("Agent run failed")
                    st.error(f"Agent failed: {e}")
                    st.stop()

            source = result.source
            if source == "Local Knowledge Base":
                color, emoji = "green", "🟢"
            elif source == "Internet Sources":
                color, emoji = "blue", "🔵"
            else:
                color, emoji = "purple", "🟣"
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{emoji} Source: {source}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(result.answer)

            with st.expander("🤖 Agent reasoning trace", expanded=False):
                for step in result.trace:
                    st.markdown(f"**{step.step}** — {step.detail}")

            if result.local_chunks:
                with st.expander(f"📚 Local evidence ({len(result.local_chunks)} chunks)", expanded=False):
                    for i, c in enumerate(result.local_chunks, 1):
                        st.markdown(
                            f"**Chunk {i}** — `{c['source']}` "
                            f"({c['modality']}, sim={c['similarity']:.2f})"
                        )
                        st.caption(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))

            if result.web_chunks:
                with st.expander(f"🌐 Web evidence ({len(result.web_chunks)} results)", expanded=False):
                    for i, r in enumerate(result.web_chunks, 1):
                        st.markdown(f"**Result {i}** — [{r.get('title', 'no title')}]({r.get('url', '#')})")
                        snippet = r.get("snippet", "") or r.get("full_text", "")
                        st.caption(snippet[:400] + ("..." if len(snippet) > 400 else ""))

            st.session_state.chat_history.append({"role": "assistant", "result": result})
