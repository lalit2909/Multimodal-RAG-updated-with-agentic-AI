# Demo Recording Script (for assignment submission)

A Loom / screen-recording walkthrough script. ~3 minutes total.

---

## Setup before recording

1. Have your terminal open in the project folder.
2. Open `https://aistudio.google.com/apikey` in another tab so you can show where the key comes from.
3. Prepare 2–3 sample files to upload:
   - A short PDF (any textbook chapter or report works)
   - An image with text on it (a poster, a chart, a slide)
   - (Optional) A short audio clip
4. Make sure `streamlit run app.py` is NOT yet running (we want to show the launch step).
5. Close other apps to keep the recording clean.

---

## Recording timeline

### 0:00 — 0:20  · Intro (face cam or voiceover)

> "Hi, this is my Enhanced Multimodal RAG system with an Agentic Architecture.
> It accepts PDF, image, audio, and video files, builds a searchable knowledge base,
> and uses an agent to decide whether to answer from local context or fall back
> to a web search — clearly labeling every answer with its source."

### 0:20 — 0:50  · Show the architecture diagram

Open `demo/index.html` in the browser. Point to the 6-step flow.

> "Here's the agentic loop: the user asks a question, the agent retrieves top-k chunks
> from the local vector store, then asks Gemini to evaluate whether those chunks are
> sufficient. If yes, it answers from local context only. If no, it runs a DuckDuckGo
> web search, combines the contexts, and generates the final answer with source
> attribution: Local, Web, or Both."

### 0:50 — 1:20  · Launch the app

Switch to terminal:

```bash
streamlit run app.py
```

> "I'm using Streamlit for the UI. Let me launch it."
> "The app opens at localhost:8501."

### 1:20 — 1:50  · Enter API key + show sidebar

- Paste your Gemini API key into the sidebar.
- Show the "Gemini client ready" success message.
- Briefly point at the Knowledge Base stats (Total chunks: 0) and the file uploader.

> "I paste my Gemini API key — it's only kept in the session, never persisted.
> The sidebar shows knowledge base stats and the file uploader for PDFs, images,
> audio, or video."

### 1:50 — 2:30  · Ingest a PDF + an image

- Drag in your sample PDF and image.
- Click **"Ingest into Knowledge Base"**.
- Watch the progress bar.
- Show the updated "Total chunks stored" metric.

> "I'll ingest a PDF and an image. The PDF goes through PyPDF2 for text extraction;
> the image goes through Gemini Vision for OCR and description. All extracted text
> is chunked, embedded with Gemini's text-embedding-004 model, and stored in ChromaDB."

### 2:30 — 3:00  · Ask a question that's IN the documents

Type a question whose answer is in your uploaded PDF.

> "This question is answerable from my local knowledge base. Notice the green badge —
> Source: Local Knowledge Base. The agent retrieved the relevant chunks, evaluated
> them as sufficient, and answered without hitting the web."

Expand the **"Agent reasoning trace"** to show the steps.

Expand **"Local evidence"** to show the retrieved chunks with similarity scores.

### 3:00 — 3:30  · Ask a question that's NOT in the documents

Type a current-events question (e.g. "What's the latest news about Mars exploration?").

> "This question isn't covered by my uploaded documents. Watch what happens — the
> agent retrieves, evaluates, finds the local context insufficient, and falls back
> to a DuckDuckGo web search. The answer is labeled blue — Source: Internet Sources."

Expand **"Web evidence"** to show the fetched URLs and snippets.

### 3:30 — 4:00  · Ask a question that draws on both

Type a question like: "Compare what's in my PDF about X with the latest information online about X."

> "For this question, the agent uses both local and web context. The answer is
> labeled purple — Source: Both — and you can see inline citations from both my
> PDF and the web sources."

### 4:00 — 4:20  · Outro

> "That's the Enhanced Multimodal RAG Agent. Source code is on GitHub — link in
> the description. Thanks for watching."

---

## Tips

- Speak slowly. Loom auto-captions help if you speak clearly.
- Hover your mouse over each UI element as you mention it.
- Keep the agent reasoning trace expanded at least once so viewers see the agentic steps.
- If the web search is slow on camera, mention "the agent is now fetching web results" while you wait.
- If you flub a line, just pause and re-say it — you can trim in post.
