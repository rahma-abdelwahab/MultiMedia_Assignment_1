import arxiv
import os
import ssl
import time
import json
import urllib.request
import shutil

# ====================== CONFIGURATION ======================

# Folder to save the downloaded PDFs
SAVE_DIR = "data/pdfs"
# Create folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# List of topics to search for
QUERIES = [
    (
        "drug_discovery",
        "cat:q-bio.BM protein structure drug target binding affinity"
    ),
]

# Number of papers to download for each topic
PAPERS_PER_TOPIC = 1

# ====================== SSL FIX ======================
# Fix for SSL certificate errors 
SSL_CTX = ssl._create_unverified_context()


def _download_pdf_ssl(url: str, out_path: str) -> None:
    """Download PDF using SSL bypass."""
    with urllib.request.urlopen(url, context=SSL_CTX) as resp, \
         open(out_path, "wb") as f:
        shutil.copyfileobj(resp, f)

# ===========================================================

def download_papers(max_per_query=10):
    """Main function to download papers from arXiv."""

    client = arxiv.Client()
    seen = set()           # Track papers we already downloaded
    mapping = {}           # Store info about each paper

    # Go through each search topic
    for category, query in QUERIES:
        print(f"\n🔍 Searching [{category}]: {query}")

        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        downloaded_this_topic = 0

        # Process each paper from the search
        for paper in client.results(search):
            # Stop when we have enough papers for this topic
            if downloaded_this_topic >= PAPERS_PER_TOPIC:
                break

            # Get the paper ID
            pid = paper.entry_id.split("/")[-1]

            # Skip if we already have this paper
            if pid in seen:
                continue
            seen.add(pid)

            # Path where the PDF will be saved
            out_path = os.path.join(SAVE_DIR, f"{pid}.pdf")

            # Save paper information
            mapping[pid] = {
                "category": category,
                "query": query,
                "title": paper.title,
                "filename": f"{pid}.pdf",
            }

            # If file already exists, skip downloading
            if os.path.exists(out_path):
                print(f" ✅ Already have {pid} — {paper.title[:60]}...")
                downloaded_this_topic += 1
                continue

            # Try to download the paper
            try:
                # Normal download using arxiv library
                paper.download_pdf(dirpath=SAVE_DIR, filename=f"{pid}.pdf")
                print(f" 📥 [{downloaded_this_topic + 1}/{PAPERS_PER_TOPIC}] Downloaded {pid}: {paper.title[:60]}...")
                downloaded_this_topic += 1
                time.sleep(1)          # Wait a bit to be nice to arXiv

            except Exception:
                # If normal download fails, try direct download with SSL fix
                try:
                    pdf_url = f"https://arxiv.org/pdf/{pid}"
                    _download_pdf_ssl(pdf_url, out_path)
                    print(f" 📥 [{downloaded_this_topic + 1}/{PAPERS_PER_TOPIC}] Downloaded {pid} (SSL bypass): {paper.title[:60]}...")
                    downloaded_this_topic += 1
                    time.sleep(1)
                except Exception as e2:
                    print(f" ❌ Failed to download {pid}: {e2}")

        print(f" → {downloaded_this_topic} paper(s) collected for [{category}]")

    # Save all paper information to a JSON file
    map_path = os.path.join(SAVE_DIR, "category_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved mapping → {map_path}")

    # Show summary table
    print(f"\n{'PDF ID':<25} {'Category':<22} {'Title'}")
    print("-" * 85)
    for pid, info in mapping.items():
        print(f"{pid:<25} {info['category']:<22} {info['title'][:50]}...")

    print(f"\n✅ Done! Total papers downloaded: {len(mapping)}")
    print(f"📁 All files saved in: {SAVE_DIR}/")

if __name__ == "__main__":
    # Run the program
    download_papers(max_per_query=15)