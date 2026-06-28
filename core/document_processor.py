"""
Multimodal document processor.

Handles ingestion of PDF, image, audio, and video files, converting each into
text chunks suitable for embedding. Uses Gemini's multimodal capabilities for
non-text formats, so we don't need separate ASR / OCR / VLM stacks.

Supported MIME types:
- PDF:    application/pdf            -> PyPDF2 text extraction; image-heavy pages fall back to Gemini
- Image:  png, jpg, jpeg, webp, gif  -> Gemini Vision description
- Audio:  wav, mp3, aac, flac, m4a   -> Gemini audio transcription
- Video:  mp4, mpeg, mov, avi, webm  -> Gemini video analysis
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

# Chunk size: roughly 800 tokens (~3200 chars). Overlap for context continuity.
CHUNK_SIZE = 3200
CHUNK_OVERLAP = 400


@dataclass
class Document:
    """A processed document ready to be embedded and stored."""
    source: str            # original filename
    modality: str          # pdf | image | audio | video | text
    chunk_id: str
    text: str
    metadata: dict


def _chunk_text(text: str, source: str, modality: str) -> List[Document]:
    """Split a long text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks: List[Document] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(
            Document(
                source=source,
                modality=modality,
                chunk_id=f"{source}__{idx}",
                text=chunk,
                metadata={
                    "source": source,
                    "modality": modality,
                    "chunk_idx": idx,
                },
            )
        )
        start = end - CHUNK_OVERLAP
        idx += 1
        if start >= len(text) - CHUNK_OVERLAP:
            break
    return chunks


# ---------- PDF ----------

def process_pdf(file_path: str, gemini: GeminiClient) -> List[Document]:
    """Extract text from a PDF using PyPDF2; fall back to Gemini for scanned/image PDFs."""
    import PyPDF2

    all_text: List[str] = []
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception as e:
                    logger.warning("Failed to extract page %d of %s: %s", i, file_path, e)
                    txt = ""
                if txt.strip():
                    all_text.append(f"--- Page {i + 1} ---\n{txt}")
                else:
                    # Image-only page -> render to image and use Gemini Vision.
                    try:
                        img = _render_pdf_page_to_image(file_path, i)
                        if img is not None:
                            desc = gemini.describe_image(
                                img,
                                prompt=(
                                    "This is page "
                                    f"{i + 1} of a PDF. Extract ALL visible text "
                                    "(OCR) and describe any figures, charts, or diagrams. "
                                    "Preserve reading order and structure."
                                ),
                            )
                            all_text.append(f"--- Page {i + 1} (OCR) ---\n{desc}")
                    except Exception as e:
                        logger.warning(
                            "Could not OCR page %d of %s: %s", i, file_path, e
                        )
    except Exception as e:
        logger.error("PDF open failed for %s: %s", file_path, e)
        raise

    full_text = "\n\n".join(all_text)
    return _chunk_text(full_text, source=os.path.basename(file_path), modality="pdf")


def _render_pdf_page_to_image(pdf_path: str, page_idx: int) -> Optional[Image.Image]:
    """Best-effort render of a PDF page to a PIL Image (used for OCR fallback)."""
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            im = page.to_image(resolution=150).original  # PIL Image
            return im
    except Exception as e:
        logger.debug("pdfplumber render unavailable: %s", e)
        return None


# ---------- Image ----------

def process_image(file_path: str, gemini: GeminiClient) -> List[Document]:
    """Open an image and ask Gemini to describe it (OCR + scene understanding)."""
    img = Image.open(file_path)
    # Convert to RGB to avoid palette/alpha issues.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    description = gemini.describe_image(img)
    text = f"[Image: {os.path.basename(file_path)}]\n{description}"
    return _chunk_text(text, source=os.path.basename(file_path), modality="image")


# ---------- Audio ----------

_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}


def process_audio(file_path: str, gemini: GeminiClient) -> List[Document]:
    """Use Gemini to transcribe + summarize audio."""
    ext = os.path.splitext(file_path)[1].lower()
    mime = _AUDIO_MIME.get(ext, "audio/mpeg")
    transcript = gemini.describe_audio(file_path, mime)
    text = f"[Audio transcript: {os.path.basename(file_path)}]\n{transcript}"
    return _chunk_text(text, source=os.path.basename(file_path), modality="audio")


# ---------- Video ----------

_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
}


def process_video(file_path: str, gemini: GeminiClient) -> List[Document]:
    """Use Gemini to analyze video (scenes + OCR + transcript)."""
    ext = os.path.splitext(file_path)[1].lower()
    mime = _VIDEO_MIME.get(ext, "video/mp4")
    summary = gemini.describe_video(file_path, mime)
    text = f"[Video analysis: {os.path.basename(file_path)}]\n{summary}"
    return _chunk_text(text, source=os.path.basename(file_path), modality="video")


# ---------- Dispatcher ----------

def process_file(file_path: str, gemini: GeminiClient) -> List[Document]:
    """Dispatch to the right processor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return process_pdf(file_path, gemini)
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
        return process_image(file_path, gemini)
    if ext in _AUDIO_MIME:
        return process_audio(file_path, gemini)
    if ext in _VIDEO_MIME:
        return process_video(file_path, gemini)
    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return _chunk_text(f.read(), source=os.path.basename(file_path), modality="text")
    raise ValueError(f"Unsupported file type: {ext}")
