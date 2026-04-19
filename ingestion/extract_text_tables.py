"""
Extract per-page text + tables from PDFs using pdfplumber.

Outputs:
- data/text_metadata.json: list of {text_id, paper_id, page_number, total_pages, text}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pdfplumber

PDF_DIR = "data/pdfs"
OUT_FILE = "data/text_metadata.json"

os.makedirs("data", exist_ok=True)


def _stringify_table(table: list[list[str | None]]) -> str:
    """Convert a table (list of rows) into a clean readable text format."""
    rows = []
    for row in table:
        # Clean -> remove None, strip spaces, replace newlines with space
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        if any(cells):   # Skip completely empty rows
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def extract_all_pdfs() -> list[dict]:
    """
    Extract text and tables from every page of all PDFs.
    Saves the result as a list of page-level entries in text_metadata.json.
    """
    items: list[dict] = []

    for pdf_path in sorted(Path(PDF_DIR).glob("*.pdf")):
        paper_id = pdf_path.stem
        print(f"Processing {paper_id} ...")
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)

                # Go through each page
                for i, page in enumerate(pdf.pages):
                    page_number = i + 1
                    text_parts: list[str] = []

                    # Extract normal text from the page
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        text_parts.append(page_text)

                    # Extract tables from the page
                    try:
                        tables = page.extract_tables() or []
                    except Exception:
                        tables = []

                    table_blocks = []
                    for t in tables:
                        s = _stringify_table(t)
                        if s:
                            table_blocks.append(s)

                    # Add tables section if any tables were found
                    if table_blocks:
                        text_parts.append("TABLES:\n" + "\n\n".join(table_blocks))
                    # Combine text and tables
                    combined = "\n\n".join(text_parts).strip()
                    # Create a unique ID for this page's text
                    text_id = f"{paper_id}_p{page_number:03d}_text"
                    # Save page information
                    items.append(
                        {
                            "text_id": text_id,
                            "paper_id": paper_id,
                            "page_number": page_number,
                            "total_pages": total_pages,
                            "text": combined,
                        }
                    )
        except Exception as e:
            print(f"  Error: {e}")
            continue

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"\nExtracted text/tables for {len(items)} pages. Metadata → {OUT_FILE}")
    return items


if __name__ == "__main__":
    
    extract_all_pdfs()