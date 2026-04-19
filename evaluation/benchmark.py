"""
Evaluation suite — tests retrieval recall + answer faithfulness
across text, table, and figure-based benchmark queries.

Recall@K is computed by checking whether ANY retrieved result
comes from the expected topic category (not a hardcoded paper ID),
making the eval robust regardless of which specific arXiv papers
were downloaded.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.retriever import retrieve
from generation.qa_chain import answer

os.makedirs("evaluation", exist_ok=True)

# ─── Benchmark definition ─────────────────────────────────────────────────────
# List of test questions to evaluate the system -> each item: (query, expected_category_keywords, modality_hint)
BENCHMARK: list[tuple[str, list[str], str]] = [
    (
        "What method is used to predict protein-ligand binding affinity?",
        ["drug", "binding", "protein", "affinity"],
        "text",
    ),
    (
        "What does the table of docking scores show across candidate compounds?",
        ["drug", "binding", "docking", "compound"],
        "table",
    ),
    (
        "Describe the trend shown in the binding affinity plot.",
        ["binding", "affinity", "protein", "drug"],
        "figure",
    ),
    (
        "What drug target is identified as most promising in the study?",
        ["drug", "target", "protein", "discovery"],
        "text",
    ),
    (
        "Which compounds show the highest selectivity in the dataset?",
        ["drug", "compound", "selectivity", "binding"],
        "table",
    ),
    (
        "What does the 3D structure visualization of the protein active site show?",
        ["protein", "structure", "active", "drug"],
        "figure",
    ),
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_category_map() -> dict[str, str]:
    """Load the category mapping from the downloaded papers."""
    map_path = "data/pdfs/category_map.json"
    try:
        with open(map_path) as f:
            raw = json.load(f)
        # Convert to {paper_id: category} format
        return {pid: info.get("category", "").lower() for pid, info in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def _is_hit(
    retrieved_papers: list[str],
    expected_keywords: list[str],
    category_map: dict[str, str],
) -> bool:
    """
    Check if any retrieved paper matches the expected topic.
    Returns True if we find a match in category or paper ID.
    """
    kws = [k.lower() for k in expected_keywords]

    for pid in retrieved_papers:
        pid_lower = pid.lower()
        cat       = category_map.get(pid, "").lower()

        # Check if any keyword appears in paper ID or category
        if any(k in pid_lower or k in cat for k in kws):
            return True

        # Extra check: split paper ID into parts (e.g. "drug_2024" → ["drug", "2024"])
        import re
        tokens = set(re.split(r"[._\-/]", pid_lower))
        # return true if keyword exactly in tokens
        if any(k in tokens for k in kws):
            return True
        # return true if keyword is part of any token
        if any(k in tok for k in kws for tok in tokens):
            return True

    return False


# ─── Main evaluation ──────────────────────────────────────────────────────────

def recall_at_k(k: int = 5) -> float:
    """Run retrieval evaluation and calculate Recall@K."""
    
    category_map = _load_category_map()

    # Show warning if category map is missing
    if not category_map:
        print(
            "⚠  WARNING: data/pdfs/category_map.json not found or empty.\n"
            "   Falling back to keyword matching against paper IDs only.\n"
            "   Add category_map.json for more accurate evaluation.\n"
        )

    hits, total = 0, 0
    results_log = []

    print(f"\n{'='*65}")
    print(f"  Retrieval Evaluation — Recall@{k}")
    print(f"{'='*65}")

    # Test each benchmark query
    for query, expected_keywords, modality in BENCHMARK:
        # Get top results from retriever
        results          = retrieve(query, top_k=k)
        retrieved_papers = [meta.get("paper_id", "") for _, meta in results]
        retrieved_scores = [score for score, _ in results]

        # Check if we got a relevant paper
        hit    = _is_hit(retrieved_papers, expected_keywords, category_map)
        # this part give me 0 or 1 according to the answer when the ansere is true it return 1 and return 0 on the false
        #  and the counter show how many questions the retriever got right.
        hits  += int(hit)
        # increasrenum of tesrequestion by 1 to show the questions were evaluated in total.
        total += 1

        icon = "✓" if hit else "✗"
        print(f"  {icon}  [{modality:6s}]  {query[:55]}")
        if retrieved_papers:
            print(f"           top-3: {retrieved_papers[:3]}")

        # Save details for the results file
        results_log.append({
            "query":             query,
            "modality":          modality,
            "expected_keywords": expected_keywords,
            "hit":               hit,
            "top_retrieved":     retrieved_papers[:3],
            "top_scores":        [round(s, 4) for s in retrieved_scores[:3]],
        })
    #  calc the final score of recall 
    recall = hits / total if total else 0.0
    print(f"\nRecall@{k}: {hits}/{total} = {recall:.1%}")

    # Save detailed results to JSON
    out_path = "evaluation/results.json"
    with open(out_path, "w") as f:
        json.dump(
            {"recall_at_k": k, "recall": recall, "details": results_log}, f, indent=2
        )
    print(f"Results saved → {out_path}")

    return recall


def faithfulness_spot_check(k: int = 3) -> None:
    """
    Run a few example questions through the full QA pipeline
    and show the generated answers for manual checking.
    """
    spot_queries = [
        "What machine learning model is used for binding affinity prediction?",
        "Summarise the findings on protein structure and drug target interaction.",
        "What does the main figure of the molecular docking results show?",
    ]

    print(f"\n{'='*65}")
    print("  Faithfulness Spot-Check (manual review)")
    print(f"{'='*65}")

    for q in spot_queries:
        print(f"\nQ: {q}")
        retrieved = retrieve(q, top_k=k)
        #  genrate answer using the QA chain
        result    = answer(q, retrieved)
        ans_preview = result["answer"].replace("\n", " ")[:400]
        print(f"A: {ans_preview}...")


if __name__ == "__main__":
    recall_at_k(k=5)
    faithfulness_spot_check(k=3)