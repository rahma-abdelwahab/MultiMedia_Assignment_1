"""
Answer generation using the Groq API with retrieved context.
Produces inline citations like [paper_id, Page X, Text].
"""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
import fitz

TEXT_META        = "data/text_metadata.json"
FIG_META         = "data/figure_metadata.json"
IMG_DIR          = "data/page_images"
MIN_DRAWING_AREA = 8_000

LLM_HOST    = os.environ.get("LLM_HOST", "https://api.groq.com/openai/v1").rstrip("/")
LLM_MODEL   = os.environ.get("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")   # set via environment variable — never hardcode


def _load_text_map() -> dict:
    """Load the text metadata file and return a dictionary for quick lookup."""
    try:
        with open(TEXT_META, encoding="utf-8") as f:
            items = json.load(f)
        return {i["text_id"]: i for i in items}
    except FileNotFoundError:
        return {}


def _load_fig_map() -> dict:
    """Load the figure metadata file and return a dictionary for quick lookup."""
    try:
        with open(FIG_META) as f:
            items = json.load(f)
        return {i["fig_id"]: i for i in items}
    except FileNotFoundError:
        return {}


def _page_has_figure(paper_id: str, page_number: int) -> bool:
    """Check if a specific page in the PDF contains a figure or drawing."""
    pdf_path = f"data/pdfs/{paper_id}.pdf"
    if not os.path.exists(pdf_path):
        return False
    try:
        doc   = fitz.open(pdf_path)
        page  = doc[page_number - 1]
        paths = page.get_drawings()
        doc.close()
        for p in paths:
            r = p.get("rect")
            if r and (r[2] - r[0]) * (r[3] - r[1]) >= MIN_DRAWING_AREA:
                return True
    except Exception:
        pass
    return False


def _build_context(snippets: list[dict]) -> str:
    """Build a clean text context from retrieved snippets to send to the LLM."""
    parts = []
    for s in snippets:
        pid  = s.get("paper_id", "?")
        page = s.get("page_number", "?")
        text = (s.get("text") or "").strip()
        cap  = (s.get("caption") or "").strip()
        mod  = s.get("modality", "image")
        if text:
            # FIX 5: increased per-snippet limit 1200 → 2000 chars
            parts.append(f"[{pid}, Page {page}, Text]\n{text[:2000]}")
        if cap:
            label = "Table" if "tbl" in s.get("item_id", "") else "Image/Chart"
            parts.append(f"[{pid}, Page {page}, {label}]\nCaption: {cap[:400]}")
        if mod == "image" and not text and not cap:
            parts.append(f"[{pid}, Page {page}, Image/Chart] (visual page — no text extracted)")
    return "\n\n---\n\n".join(parts) if parts else "No context retrieved."


def _call_llm(
    query: str,
    context: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str]:
    """"Call the LLM (Large Language Model) to generate an answer.""""
    host = (base_url or LLM_HOST).rstrip("/")
    m    = model   or LLM_MODEL
    key  = api_key or LLM_API_KEY
# Make sure we have an API key
    if not key:
        raise RuntimeError(
            "No API key found. Set the LLM_API_KEY environment variable:\n"
            "  PowerShell: $env:LLM_API_KEY='your_key_here'\n"
            "  Linux/Mac:  export LLM_API_KEY='your_key_here'"
        )

    system = (
        "You are a research assistant. Answer the question using ONLY the retrieved context. "
        "After every sentence add an inline citation: [paper_id, Page X, Text] or "
        "[paper_id, Page X, Image/Chart]. Be concise and accurate."
    )
    # Prepare the user message
    user_content = f"Context:\n\n{context[:6000]}\n\nQuestion: {query}"

    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "stream":     False,

        "max_tokens": 1024,
    }

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
# Create the HTTP request
    req = urllib.request.Request(
        f"{host}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        # Send the request to the LLM API
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # Handle HTTP errors
        body = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"LLM API HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        # Handle connection errors
        raise RuntimeError(
            f"{e.reason!s} — Is the LLM service reachable?"
        ) from e

    data    = json.loads(raw)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"LLM returned an empty reply: {raw[:500]}")

    msg         = choices[0].get("message") or {}
    answer_text = (msg.get("content") or "").strip()
    if not answer_text:
        raise RuntimeError(f"LLM returned empty content: {raw[:500]}")

    return answer_text, data.get("model", m)


def answer(
    query: str,
    retrieved_items: list,
    *,
    llm_host:    str | None = None,
    llm_model:   str | None = None,
    llm_api_key: str | None = None,
) -> dict:
"""Generate a final answer using the LLM based on the retrieved items."""

# Load metadata for text chunks and figures
    text_map = _load_text_map()
    fig_map  = _load_fig_map()

    snippets: list[dict] = []

    # Take only the top 5 retrieved items and prepare them for the LLM
    for score, p in retrieved_items[:5]:
        item_id      = p.get("item_id", "")
        modality     = p.get("modality", "image")
        content_type = p.get("content_type", "page")
        paper_id     = p.get("paper_id", "")
        page_number  = p.get("page_number", 0)
        image_path   = p.get("image_path", "")
        caption      = p.get("caption", "")

        if not image_path or not os.path.exists(image_path):
            image_path = f"{IMG_DIR}/{paper_id}_p{page_number:03d}.jpg"
# Try to get the extracted text for this page or item
        tid  = f"{paper_id}_p{page_number:03d}_text"
        text = (
            text_map.get(tid,      {}).get("text") or
            text_map.get(item_id,  {}).get("text") or ""
        ).strip()

        is_figure_crop = (content_type in ("figure", "table")) and (item_id in fig_map)
        has_figure     = is_figure_crop or _page_has_figure(paper_id, page_number)

        snippets.append({
            "item_id":        item_id,
            "modality":       modality,
            "content_type":   content_type,
            "paper_id":       paper_id,
            "page_number":    page_number,
            "image_path":     image_path,
            "text":           text,
            "caption":        caption,
            "_score":         score,
            "is_figure_crop": is_figure_crop,
            "has_figure":     has_figure,
        })

    if not snippets:
        return {"answer": "No relevant content found.", "snippets": [], "model": "none"}

    try:
        # Build context string from the snippet
        context      = _build_context(snippets)
        # Call the LLM to generate the answer
        ans, model   = _call_llm(
            query, context,
            base_url=llm_host,
            model=llm_model,
            api_key=llm_api_key,
        )
    except Exception as exc:
        # if llm fail show this answer 
        ans   = f"⚠️ LLM API error: {exc}\n\nShowing raw retrieved evidence below."
        model = "error"

    return {"answer": ans, "snippets": snippets, "model": model}