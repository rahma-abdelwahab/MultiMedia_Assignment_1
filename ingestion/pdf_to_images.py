import fitz
from pathlib import Path
import json, os

PDF_DIR   = "data/pdfs"
IMG_DIR   = "data/page_images"
META_FILE = "data/page_metadata.json"

os.makedirs(IMG_DIR, exist_ok=True) 

def ingest_all_pdfs(dpi=300):
    metadata = [] 
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # scale factor to control image resolution 

    for pdf_path in sorted(Path(PDF_DIR).glob("*.pdf")):  
        paper_id = pdf_path.stem  # get file name without extension
        print(f"Processing {paper_id} ...")
        try:
            doc = fitz.open(str(pdf_path))  
        except Exception as e:
            print(f"  Error opening: {e}")  
            continue  

        total_pages = len(doc)
        for i, page in enumerate(doc):
            page_id  = f"{paper_id}_p{i+1:03d}"
            img_path = os.path.join(IMG_DIR, f"{page_id}.jpg")
            try:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)  # convert page to image
                pix.save(img_path)
                metadata.append({
                    "page_id":     page_id,
                    "paper_id":    paper_id,
                    "page_number": i + 1,
                    "total_pages": total_pages,
                    "image_path":  img_path,
                })
                print(f"  page {i+1}/{total_pages} → {img_path}")
            except Exception as e:
                print(f"  Error page {i+1}: {e}")
        doc.close()

    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nIngested {len(metadata)} pages. Metadata → {META_FILE}")
    return metadata

if __name__ == "__main__":
    ingest_all_pdfs()