# How to push this project to GitHub

**Important**: Never paste your GitHub Personal Access Token in plain text anywhere — not in chat, not in commits, not in config files. The steps below use Git's secure credential cache so the token is stored locally on your machine only.

---

## Step 1 — Revoke the token you leaked

Go to https://github.com/settings/tokens and **delete** the token you pasted in chat. Generate a fresh one with the `repo` scope. Don't share it with anyone — including me.

---

## Step 2 — Create the empty repo on GitHub

1. Go to https://github.com/new
2. Repository name: `enhanced-multimodal-rag-agent`
3. Set to **Public** (so your instructor can view it without being added as a collaborator)
4. **Do NOT** check "Initialize this repository with a README" — your local repo already has commits
5. Click **Create repository**

---

## Step 3 — From your local machine, link and push

On whatever machine you have this project folder on:

```bash
cd /path/to/enhanced-multimodal-rag-agent

# Add the GitHub remote (replace lalit2029 with your username if different)
git remote add origin https://github.com/lalit2029/enhanced-multimodal-rag-agent.git

# Push (this will prompt for username + password — use your token as the password)
git push -u origin main
```

When prompted:
- **Username**: `lalit2029`
- **Password**: paste your **new** Personal Access Token (not your account password)

Git will remember the credentials for the session. To make it remember across reboots on Linux/macOS:

```bash
git config --global credential.helper store
```

(Or use `osxkeychain` on macOS, `manager` on Windows.)

---

## Step 4 — Enable GitHub Pages for the demo page

This gives you a public URL you can paste into your assignment submission as a "preview":

1. Go to your repo: `https://github.com/lalit2029/enhanced-multimodal-rag-agent`
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: `main` / folder: `/demo`
   - Click **Save**
4. Wait ~1 minute. Your demo page will be live at:

   ```
   https://lalit2029.github.io/enhanced-multimodal-rag-agent/
   ```

This is a static HTML page that shows the architecture diagram, source-badge legend, tech stack, and sample chat. The actual functional Streamlit app must still be run locally with `streamlit run app.py`.

---

## Step 5 — Record your demo

Follow `demo/RECORDING_SCRIPT.md` — it's a ~4-minute Loom-style walkthrough that demonstrates:
1. Architecture overview
2. App launch
3. API key entry + file upload + ingestion
4. Three queries that trigger each path:
   - Local KB only (green badge)
   - Web search fallback (blue badge)
   - Combined sources (purple badge)

---

## Quick verification checklist

After pushing, your repo should have these files visible on GitHub:

```
enhanced-multimodal-rag-agent/
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── gemini_client.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── web_search.py
│   └── agent.py
├── data/
│   ├── uploads/.gitkeep
│   └── chroma_db/.gitkeep
└── demo/
    ├── index.html
    └── RECORDING_SCRIPT.md
```

That's it. You're done. 🎉
