"""
Extract figures, charts, mathematical equations, and tables as cropped images from PDFs.
Saves crops to data/figure_images/ and metadata to data/figure_metadata.json
"""
import fitz
import pdfplumber
import json
import os
import re
from pathlib import Path

PDF_DIR     = "data/pdfs"
FIG_DIR     = "data/figure_images"
FIG_META    = "data/figure_metadata.json"

os.makedirs(FIG_DIR, exist_ok=True)

def _extract_visuals_pymupdf(doc, paper_id, page_idx, dpi=300):
    """Extract real embedded figures and mathematical equations."""
    page     = doc[page_idx]
    page_num = page_idx + 1
    mat      = fitz.Matrix(dpi / 72, dpi / 72)
    crops    = []

    # ==========================================
    # 1. EXTRACT EMBEDDED FIGURES & CHARTS
    # ==========================================
    # get_image_info grabs actual images/plots embedded in the PDF (much safer than get_drawings)
    images = page.get_image_info(xrefs=True)
    
    for img_idx, img in enumerate(images):
        bbox = fitz.Rect(img["bbox"])
        w, h = bbox.width, bbox.height
        
        # Filter out tiny logos, noise, or 1x1 pixel artifacts
        if w * h < 4000:
            continue

        # Add generous padding to capture axes labels and context
        padded = fitz.Rect(
            max(0, bbox.x0 - 20),
            max(0, bbox.y0 - 20),
            min(page.rect.width,  bbox.x1 + 20),
            min(page.rect.height, bbox.y1 + 20),
        )

        fig_id   = f"{paper_id}_p{page_num:03d}_fig{img_idx+1:02d}"
        img_path = os.path.join(FIG_DIR, f"{fig_id}.jpg")

        clip = page.get_pixmap(matrix=mat, clip=padded, colorspace=fitz.csRGB)
        clip.save(img_path)

        crops.append({
            "fig_id":      fig_id,
            "paper_id":    paper_id,
            "page_number": page_num,
            "type":        "figure",
            "image_path":  img_path,
            "bbox":        [padded.x0, padded.y0, padded.x1, padded.y1],
            "caption":     "", 
        })

    # ==========================================
    # 2. EXTRACT MATHEMATICAL EQUATIONS
    # ==========================================
    blocks = page.get_text("blocks")
    math_idx = 0
    math_symbols = ['∑', '∫', '≈', '±', '≤', '≥', '∝', '∂', 'Δ', 'θ', 'λ', 'μ', 'π', 'σ', 'ω', '=']
    
    for b in blocks:
        # b is (x0, y0, x1, y1, text, block_no, block_type)
        if b[6] != 0: # 0 means text block, skip images/drawings
            continue
            
        text = b[4].strip()
        is_math = False
        
        # Heuristic 1: Block ends with an equation number like (1), (2a), etc.
        if re.search(r'\(\d+[a-zA-Z]*\)$', text):
            is_math = True
        # Heuristic 2: Short blocks containing heavy math symbols
        elif any(sym in text for sym in math_symbols) and len(text.split()) < 20:
            is_math = True

        if is_math:
            bbox = fitz.Rect(b[:4])

            # Pad the equation so symbols aren't cut off
            padded = fitz.Rect(
                max(0, bbox.x0 - 15),
                max(0, bbox.y0 - 15),
                min(page.rect.width,  bbox.x1 + 15),
                min(page.rect.height, bbox.y1 + 15),
            )

            math_idx += 1
            fig_id   = f"{paper_id}_p{page_num:03d}_math{math_idx:02d}"
            img_path = os.path.join(FIG_DIR, f"{fig_id}.jpg")

            clip = page.get_pixmap(matrix=mat, clip=padded, colorspace=fitz.csRGB)
            clip.save(img_path)

            crops.append({
                "fig_id":      fig_id,
                "paper_id":    paper_id,
                "page_number": page_num,
                "type":        "equation", # Tagged as equation!
                "image_path":  img_path,
                "bbox":        [padded.x0, padded.y0, padded.x1, padded.y1],
                "caption":     text,
            })

    return crops


def _extract_tables_pdfplumber(pdf_path, paper_id, doc_fitz, dpi=300):
    """Extract table regions as cropped images + their text content."""
    crops = []
    mat   = fitz.Matrix(dpi / 72, dpi / 72)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num  = page_idx + 1
            fitz_page = doc_fitz[page_idx]

            try:
                tables = page.find_tables()
            except Exception:
                continue

            for tbl_idx, tbl in enumerate(tables):
                bbox = tbl.bbox

                fitz_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                padded    = fitz.Rect(
                    max(0, fitz_rect.x0 - 10),
                    max(0, fitz_rect.y0 - 10),
                    min(fitz_page.rect.width,  fitz_rect.x1 + 10),
                    min(fitz_page.rect.height, fitz_rect.y1 + 10),
                )

                fig_id   = f"{paper_id}_p{page_num:03d}_tbl{tbl_idx+1:02d}"
                img_path = os.path.join(FIG_DIR, f"{fig_id}.jpg")

                clip = fitz_page.get_pixmap(matrix=mat, clip=padded, colorspace=fitz.csRGB)
                clip.save(img_path)

                try:
                    rows = tbl.extract()
                    text_rows = []
                    for row in (rows or []):
                        cells = [(c or "").strip() for c in row]
                        if any(cells):
                            text_rows.append(" | ".join(cells))
                    table_text = "\n".join(text_rows)
                except Exception:
                    table_text = ""

                crops.append({
                    "fig_id":      fig_id,
                    "paper_id":    paper_id,
                    "page_number": page_num,
                    "type":        "table",
                    "image_path":  img_path,
                    "bbox":        list(bbox),
                    "caption":     table_text[:400],
                })

    return crops


def extract_all():
    all_figures = []

    for pdf_path in sorted(Path(PDF_DIR).glob("*.pdf")):
        paper_id = pdf_path.stem
        if paper_id == "category_map":
            continue
        print(f"\nProcessing {paper_id} ...")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # 1. Extract figures and math equations
        for page_idx in range(len(doc)):
            figs = _extract_visuals_pymupdf(doc, paper_id, page_idx)
            all_figures.extend(figs)
            if figs:
                print(f"  page {page_idx+1}: {len(figs)} figure/math crop(s)")

        # 2. Extract tables
        tbl_crops = _extract_tables_pdfplumber(pdf_path, paper_id, doc)
        all_figures.extend(tbl_crops)
        if tbl_crops:
            print(f"  {len(tbl_crops)} table crop(s) across all pages")

        doc.close()

    with open(FIG_META, "w") as f:
        json.dump(all_figures, f, indent=2)

    print(f"\nDone. {len(all_figures)} figure/table/math crops → {FIG_META}")
    print(f"Images saved to {FIG_DIR}/")
    return all_figures


if __name__ == "__main__":
    extract_all()