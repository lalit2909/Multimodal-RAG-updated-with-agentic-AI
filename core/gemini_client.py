"""
Gemini API client wrapper (using the new google-genai SDK).

Centralizes all Gemini API calls:
- LLM text generation (gemini-2.0-flash)
- Embeddings (text-embedding-004)
- Multimodal content understanding (images / audio / video via gemini-2.0-flash)

The API key is passed in at construction time (read from Streamlit session state
in app.py); nothing is hardcoded.
"""
from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)

# Model names - pinned for reproducibility.
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"


class GeminiClient:
    """Thin wrapper around the google-genai SDK."""

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required.")
        self._client = genai.Client(api_key=api_key.strip())
        self._api_key = api_key.strip()

    # ---------- Text generation ----------

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a text response. Optionally prepend retrieved context."""
        parts: List[str] = []
        if system_instruction:
            parts.append(f"SYSTEM:\n{system_instruction}\n\n")
        if context:
            parts.append(f"CONTEXT:\n{context}\n\n")
        parts.append(f"USER QUERY:\n{prompt}\n\nAnswer:")
        full_prompt = "\n".join(parts)

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                ],
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = self._client.models.generate_content(
                model=LLM_MODEL,
                contents=full_prompt,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as e:
            logger.error("Gemini generate failed: %s", e)
            raise RuntimeError(f"Gemini generation failed: {e}") from e

    # ---------- Embeddings ----------

    def embed(self, text: str) -> List[float]:
        """Generate a dense embedding vector for a single text chunk."""
        try:
            result = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return list(result.embeddings[0].values)
        except Exception as e:
            logger.error("Gemini embed failed: %s", e)
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in one call (Gemini supports batch input)."""
        if not texts:
            return []
        try:
            result = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [list(e.values) for e in result.embeddings]
        except Exception as e:
            logger.error("Gemini embed_batch failed; falling back to per-item. %s", e)
            return [self.embed(t) for t in texts]

    # ---------- Multimodal understanding ----------

    def describe_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Use Gemini Vision to convert an image into a textual description."""
        instruction = prompt or (
            "Describe this image in detail. Extract all visible text (OCR), "
            "identify objects, people, scenes, charts, diagrams, and any other "
            "notable content. Be thorough but factual."
        )
        try:
            response = self._client.models.generate_content(
                model=LLM_MODEL,
                contents=[instruction, image],
            )
            return (response.text or "").strip()
        except Exception as e:
            logger.error("Image description failed: %s", e)
            raise RuntimeError(f"Image understanding failed: {e}") from e

    def describe_audio(self, audio_path: str, mime_type: str) -> str:
        """Upload an audio file and ask Gemini to transcribe + summarize."""
        try:
            audio_file = self._client.files.upload(file=audio_path)
            response = self._client.models.generate_content(
                model=LLM_MODEL,
                contents=[
                    "Transcribe this audio verbatim, then provide a concise summary "
                    "of the key points. If there are multiple speakers, label them. "
                    "If the audio is music, describe genre, mood, and any lyrics.",
                    audio_file,
                ],
            )
            text = (response.text or "").strip()
            try:
                self._client.files.delete(name=audio_file.name)
            except Exception:
                pass
            return text
        except Exception as e:
            logger.error("Audio description failed: %s", e)
            raise RuntimeError(f"Audio understanding failed: {e}") from e

    def describe_video(self, video_path: str, mime_type: str) -> str:
        """Upload a video file and ask Gemini to summarize content + speech."""
        try:
            video_file = self._client.files.upload(file=video_path)
            # Gemini needs the file to finish processing before it can be used.
            while video_file.state and video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = self._client.files.get(name=video_file.name)

            response = self._client.models.generate_content(
                model=LLM_MODEL,
                contents=[
                    "Analyze this video. Provide: (1) a scene-by-scene summary, "
                    "(2) any visible on-screen text (OCR), (3) a transcript of "
                    "spoken content if present, (4) overall topic and key takeaways.",
                    video_file,
                ],
            )
            text = (response.text or "").strip()
            try:
                self._client.files.delete(name=video_file.name)
            except Exception:
                pass
            return text
        except Exception as e:
            logger.error("Video description failed: %s", e)
            raise RuntimeError(f"Video understanding failed: {e}") from e


# Confidence-evaluation prompt for the agentic layer.
EVALUATE_CONTEXT_PROMPT = """You are a relevance evaluator for a retrieval-augmented generation system.

Given a user query and a set of retrieved context chunks (each labeled [LOCAL] or [WEB]),
decide whether the context contains ENOUGH information to answer the query accurately.

Respond with EXACTLY one JSON object on a single line, no markdown, no explanation:
{"sufficient": true|false, "confidence": 0.0-1.0, "reason": "one short sentence"}

Definitions:
- "sufficient": true if the context lets you write a factually grounded answer with no major gaps.
- "confidence": your confidence in that judgment.
- Only use [WEB] context as supplementary; if [LOCAL] alone is enough, still mark sufficient=true.
"""


def evaluate_context(client: GeminiClient, query: str, context_chunks: List[str]) -> dict:
    """Ask Gemini to judge whether retrieved context is sufficient."""
    context_block = "\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    prompt = (
        f"{EVALUATE_CONTEXT_PROMPT}\n\n"
        f"USER QUERY: {query}\n\n"
        f"CONTEXT CHUNKS:\n{context_block}\n\nJSON:"
    )
    raw = client.generate(prompt, temperature=0.0, max_tokens=200)
    try:
        cleaned = raw.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception:
        # Default to "insufficient" if parsing fails - safer to trigger web search.
        return {"sufficient": False, "confidence": 0.0, "reason": "parse failure"}
