import arxiv
import os
import time
import json

# ====================== CONFIGURATION ======================

# Directory where downloaded PDF files will be saved
SAVE_DIR = "data/pdfs"

# Create the directory if it doesn't exist (no error if it already exists)
os.makedirs(SAVE_DIR, exist_ok=True)

# List of search topics
# Format: ("category_name", "arxiv_search_query")
QUERIES = [
    (
        "drug_discovery",
        "cat:q-bio.BM protein structure drug target binding affinity"
    ),
    # Example of another topic (currently disabled):
    # (
    #     "clinical_outcomes",
    #     "cat:q-bio.QM clinical trial treatment effect mortality survival"
    # ),
]

# How many papers to download per topic
PAPERS_PER_TOPIC = 1

# ===========================================================

def download_papers(max_per_query=10):
    """
    Downloads research papers from arXiv based on the defined queries.
    
    Args:
        max_per_query (int): Maximum number of search results to fetch per query.
                            Default is 10.
    
    The function:
    - Searches arXiv for each topic
    - Downloads PDFs (skips if already downloaded)
    - Saves metadata in a JSON file for easy tracking
    - Prints a summary table at the end
    """
    
    # Initialize arXiv client
    client = arxiv.Client()
    
    # Set to track already processed paper IDs (to avoid duplicates)
    seen = set()
    
    # Dictionary to store metadata for all downloaded papers
    mapping = {}

    # Loop through each topic/category
    for category, query in QUERIES:
        print(f"\n🔍 Searching [{category}]: {query}")
        
        # Create search object
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.Relevance,   # Sort by most relevant first
        )

        downloaded_this_topic = 0

        # Iterate over search results
        for paper in client.results(search):
            
            # Stop if we reached the desired number of papers for this topic
            if downloaded_this_topic >= PAPERS_PER_TOPIC:
                break

            # Extract paper ID (e.g., 2504.12345)
            pid = paper.entry_id.split("/")[-1]

            # Skip if we already downloaded this paper
            if pid in seen:
                continue
            seen.add(pid)

            # Define output path for the PDF
            out_path = os.path.join(SAVE_DIR, f"{pid}.pdf")

            # Store paper metadata in mapping dictionary
            mapping[pid] = {
                "category": category,
                "query": query,
                "title": paper.title,
                "filename": f"{pid}.pdf",
            }

            # Check if file already exists
            if os.path.exists(out_path):
                print(f"  ✅ Already have {pid} — {paper.title[:60]}...")
                downloaded_this_topic += 1
                continue

            # Download the paper
            try:
                paper.download_pdf(dirpath=SAVE_DIR, filename=f"{pid}.pdf")
                print(f"  📥 [{downloaded_this_topic + 1}/{PAPERS_PER_TOPIC}] Downloaded {pid}: {paper.title[:60]}...")
                downloaded_this_topic += 1
                
                # Be respectful to arXiv servers - add small delay
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Failed to download {pid}: {e}")

        print(f"  → {downloaded_this_topic} paper(s) collected for [{category}]")

    # ====================== SAVE METADATA ======================
    
    # Save mapping to JSON file (very useful for later processing)
    map_path = os.path.join(SAVE_DIR, "category_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved mapping → {map_path}")

    # ====================== PRINT SUMMARY TABLE ======================
    
    print(f"\n{'PDF ID':<25} {'Category':<22} {'Title'}")
    print("-" * 85)
    
    for pid, info in mapping.items():
        print(f"{pid:<25} {info['category']:<22} {info['title'][:50]}...")

    print(f"\n✅ Done! Total papers downloaded: {len(mapping)}")
    print(f"📁 All files saved in: {SAVE_DIR}/")


# ====================== RUN THE SCRIPT ======================

if __name__ == "__main__":
    # You can change the number of results to search here if needed
    download_papers(max_per_query=15)