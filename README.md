# Enhanced Multimodal RAG with Agentic Architecture

An end-to-end multimodal Retrieval-Augmented Generation system with an **agentic decision layer** that chooses between a local knowledge base and web search, and clearly labels every answer with its source.

## What it does

1. User submits a query
2. The agent retrieves the top-k most relevant chunks from the **local knowledge base** (ChromaDB)
3. The agent uses Gemini to **evaluate** whether those chunks are sufficient
4. **If sufficient** → answer is generated from local context only → labeled `🟢 Local Knowledge Base`
5. **If insufficient** → the agent runs a **DuckDuckGo web search**, fetches the top results, and combines local + web context
6. The final answer is labeled with one of three source badges:
   - 🟢 **Local Knowledge Base** — answered entirely from your uploaded files
   - 🔵 **Internet Sources** — local KB didn't have it, fell back to the web
   - 🟣 **Both** — answer materially drew on local + web context

## Multimodal ingestion

| Modality | Formats | How it's processed |
|---|---|---|
| PDF | `.pdf` | PyPDF2 text extraction; image-only pages fall back to Gemini Vision OCR |
| Image | `.png .jpg .jpeg .webp .gif .bmp` | Gemini Vision: OCR + scene description |
| Audio | `.wav .mp3 .aac .flac .m4a .ogg` | Gemini native audio: transcript + summary |
| Video | `.mp4 .mpeg .mov .avi .webm` | Gemini native video: scene summary + OCR + transcript |
| Text | `.txt .md` | Read directly |

All extracted text is chunked (~800 tokens with overlap) and embedded with Gemini's `text-embedding-004` model into a persistent ChromaDB store.

## Tech stack

- **Frontend**: Streamlit
- **LLM + embeddings + multimodal**: Google Gemini API (`gemini-2.0-flash`, `text-embedding-004`)
- **Vector DB**: ChromaDB (persistent, on-disk)
- **Web search**: DuckDuckGo (no API key required)
- **PDF parsing**: PyPDF2 + pdfplumber (OCR fallback)

## Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/enhanced-multimodal-rag-agent.git
cd enhanced-multimodal-rag-agent
python -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a Gemini API key

Visit https://aistudio.google.com/apikey and create a free key.

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 4. Use it

1. Paste your Gemini API key in the sidebar
2. Upload PDF / image / audio / video files and click **Ingest into Knowledge Base**
3. Ask questions in the chat box
4. Every answer shows a colored source badge (🟢 / 🔵 / 🟣) and an expandable **Agent reasoning trace** panel that walks through each step the agent took

## Project structure

```
.
├── app.py                       # Streamlit UI
├── requirements.txt
├── README.md
├── core/
│   ├── gemini_client.py         # Gemini LLM + embeddings + multimodal wrapper
│   ├── document_processor.py    # PDF / image / audio / video ingestion
│   ├── vector_store.py          # ChromaDB persistence + retrieval
│   ├── web_search.py            # DuckDuckGo search + URL fetch
│   └── agent.py                 # Agentic decision loop
├── data/
│   ├── uploads/                 # (temp files during ingestion)
│   └── chroma_db/               # Persistent vector store
└── demo/
    └── index.html               # Static HTML demo (optional GitHub Pages preview)
```

## Agentic decision logic

The agent follows this loop (see `core/agent.py`):

1. **Retrieve** top-5 chunks from the local KB (cosine similarity ≥ 0.30)
2. **Evaluate** sufficiency via Gemini — returns `{sufficient, confidence, reason}`
3. **Decision**:
   - If `sufficient && confidence ≥ 0.55` → answer from local KB only
   - Else → run DuckDuckGo web search, fetch full text for top-5 results, re-evaluate combined context, then generate the final answer
4. **Source attribution**: scan the final answer for inline citations / source filenames / URLs and label `Local / Web / Both` accordingly

## Tuning knobs

In `core/agent.py`:

| Constant | Default | Purpose |
|---|---|---|
| `LOCAL_TOP_K` | 5 | How many local chunks to retrieve |
| `WEB_MAX_RESULTS` | 5 | How many DuckDuckGo results to fetch |
| `SIMILARITY_FLOOR` | 0.30 | Below this, chunks are treated as noise |
| `SUFFICIENCY_CONFIDENCE_FLOOR` | 0.55 | Below this confidence, fall back to web |

## Demo recording script

A Loom-style walkthrough script is included in `demo/RECORDING_SCRIPT.md` — read it aloud while recording your screen for a clean assignment submission.

## Notes

- No LLM runs locally; everything goes through the Gemini API.
- Your Gemini API key is never persisted — it lives only in the Streamlit session.
- The vector DB persists across restarts in `data/chroma_db/`.

## License

MIT — see `LICENSE` (add one if you want).
