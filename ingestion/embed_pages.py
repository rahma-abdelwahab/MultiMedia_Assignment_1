"""
Embed page images, text pages, and figure/table crops using ColPali.
Embeds ALL papers found in data/pdfs/ (no hard-coded selection).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

META_FILE  = "data/page_metadata.json"
TEXT_META  = "data/text_metadata.json"
FIG_META   = "data/figure_metadata.json"
EMBED_FILE = "data/embeddings.pt"
PDF_DIR    = "data/pdfs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ─── model loading ──────────────────────────────────────────────────────────

def load_model() -> tuple[ColPali, ColPaliProcessor]:
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    return model, processor


# ─── embedding helpers ───────────────────────────────────────────────────────

def _embed_images(
    model: ColPali,
    processor: ColPaliProcessor,
    entries: list[dict],
    batch_size: int,
    all_embeddings: dict,
) -> None:
    """Embed full-page images."""
    valid   = [e for e in entries if os.path.exists(e["image_path"])]
    images  = [Image.open(e["image_path"]).convert("RGB") for e in valid]
    ids     = [e["page_id"] for e in valid]

    print(f"Embedding {len(images)} page images ...")
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_ids  = ids[i : i + batch_size]
        with torch.no_grad():
            inputs = processor.process_images(batch_imgs).to(DEVICE)
            embs   = model(**inputs)
        for j, pid in enumerate(batch_ids):
            all_embeddings[pid] = embs[j].cpu()
        print(f"  page images {min(i + batch_size, len(images))}/{len(images)} done")


def _embed_text(
    model: ColPali,
    processor: ColPaliProcessor,
    text_entries: list[dict],
    batch_size: int,
    all_embeddings: dict,
) -> None:
    """Embed text / table pages."""
    filtered = [
        (e["text_id"], (e.get("text") or "").strip())
        for e in text_entries
        if (e.get("text") or "").strip()
    ]
    if not filtered:
        print("No non-empty text pages to embed.")
        return

    ids, texts = zip(*filtered)
    ids, texts = list(ids), list(texts)
    print(f"Embedding {len(texts)} text/table pages ...")
    for i in range(0, len(texts), batch_size):
        batch_txt = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        with torch.no_grad():
            inputs = processor.process_queries(batch_txt).to(DEVICE)
            embs   = model(**inputs)
        for j, tid in enumerate(batch_ids):
            all_embeddings[tid] = embs[j].cpu()
        print(f"  text {min(i + batch_size, len(texts))}/{len(texts)} done")


def _embed_figures(
    model: ColPali,
    processor: ColPaliProcessor,
    fig_entries: list[dict],
    batch_size: int,
    all_embeddings: dict,
) -> None:
    """Embed figure / table crop images."""
    entries = [e for e in fig_entries if os.path.exists(e["image_path"])]
    if not entries:
        print("No figure crops to embed.")
        return

    images = [Image.open(e["image_path"]).convert("RGB") for e in entries]
    ids    = [e["fig_id"] for e in entries]

    print(f"Embedding {len(images)} figure/table crops ...")
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_ids  = ids[i : i + batch_size]
        with torch.no_grad():
            inputs = processor.process_images(batch_imgs).to(DEVICE)
            embs   = model(**inputs)
        for j, fid in enumerate(batch_ids):
            all_embeddings[fid] = embs[j].cpu()
        print(f"  figures {min(i + batch_size, len(images))}/{len(images)} done")


# ─── main entry point ────────────────────────────────────────────────────────

def embed_all(batch_size: int = 2) -> None:
    """
    Embed every paper that has been ingested (page images + text + figures).
    The paper list is derived automatically from data/page_metadata.json —
    no hard-coded paper IDs.
    """
    # Load all metadata
    with open(META_FILE) as f:
        all_page_meta: list[dict] = json.load(f)

    try:
        with open(TEXT_META, encoding="utf-8") as f:
            all_text_meta: list[dict] = json.load(f)
    except FileNotFoundError:
        all_text_meta = []

    try:
        with open(FIG_META) as f:
            all_fig_meta: list[dict] = json.load(f)
    except FileNotFoundError:
        all_fig_meta = []

    # Discover available paper IDs from the PDF directory
    available_papers = {p.stem for p in Path(PDF_DIR).glob("*.pdf")}
    print(f"Found {len(available_papers)} paper(s) in {PDF_DIR}/")

    # Filter metadata to only available papers
    page_metadata = [e for e in all_page_meta if e["paper_id"] in available_papers]
    text_metadata = [e for e in all_text_meta if e["paper_id"] in available_papers]
    fig_metadata  = [e for e in all_fig_meta  if e["paper_id"] in available_papers]

    print(
        f"Embedding:\n"
        f"  {len(page_metadata)} page images\n"
        f"  {len(text_metadata)} text pages\n"
        f"  {len(fig_metadata)}  figure/table crops"
    )

    if not page_metadata:
        print("ERROR: No page metadata found. Run pdf_to_images.py first.")
        return

    # Load existing embeddings (incremental — skip already embedded items)
    if os.path.exists(EMBED_FILE):
        print(f"Loading existing embeddings from {EMBED_FILE} ...")
        all_embeddings: dict = torch.load(EMBED_FILE, weights_only=False)
        print(f"  {len(all_embeddings)} already embedded.")
    else:
        all_embeddings = {}

    # Filter to only new items
    page_metadata = [e for e in page_metadata if e["page_id"] not in all_embeddings]
    text_metadata = [e for e in text_metadata if e["text_id"] not in all_embeddings]
    fig_metadata  = [e for e in fig_metadata  if e["fig_id"]  not in all_embeddings]

    if not any([page_metadata, text_metadata, fig_metadata]):
        print("All items already embedded. Nothing to do.")
        return

    model, processor = load_model()

    if page_metadata:
        _embed_images(model, processor, page_metadata, batch_size, all_embeddings)
    if text_metadata:
        _embed_text(model, processor, text_metadata, batch_size, all_embeddings)
    if fig_metadata:
        _embed_figures(model, processor, fig_metadata, batch_size, all_embeddings)

    torch.save(all_embeddings, EMBED_FILE)
    print(f"\nDone. Embeddings saved → {EMBED_FILE} (total: {len(all_embeddings)} items)")


if __name__ == "__main__":
    embed_all()