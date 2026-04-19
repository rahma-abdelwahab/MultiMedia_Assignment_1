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

# ─── Benchmark definition ────────────────────────────────────────────────────
# Each entry: (query, expected_category_keywords, modality_hint)
# expected_category_keywords: list of strings; a hit is recorded when ANY
# retrieved paper's category OR paper_id contains at least one keyword.
# Topic: drug_discovery (cat:q-bio.BM — protein structure, drug targets, binding)
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_category_map() -> dict[str, str]:
    """Return {paper_id: category} from the download mapping file."""
    map_path = "data/pdfs/category_map.json"
    try:
        with open(map_path) as f:
            raw = json.load(f)
        return {pid: info.get("category", "").lower() for pid, info in raw.items()}
    except FileNotFoundError:
        return {}


def _is_hit(
    retrieved_papers: list[str],
    expected_keywords: list[str],
    category_map: dict[str, str],
) -> bool:
    """
    A hit if any retrieved paper's category or paper_id contains
    at least one of the expected keywords (case-insensitive).
    """
    kws = [k.lower() for k in expected_keywords]
    for pid in retrieved_papers:
        pid_lower = pid.lower()
        cat       = category_map.get(pid, "").lower()
        if any(k in pid_lower or k in cat for k in kws):
            return True
    return False


# ─── Main evaluation ─────────────────────────────────────────────────────────

def recall_at_k(k: int = 5) -> float:
    category_map  = _load_category_map()
    hits, total   = 0, 0
    results_log   = []

    print(f"\n{'='*65}")
    print(f"  Retrieval Evaluation — Recall@{k}")
    print(f"{'='*65}")

    for query, expected_keywords, modality in BENCHMARK:
        results           = retrieve(query, top_k=k)
        retrieved_papers  = [meta.get("paper_id", "") for _, meta in results]
        retrieved_scores  = [score for score, _ in results]

        hit = _is_hit(retrieved_papers, expected_keywords, category_map)
        hits  += int(hit)
        total += 1

        icon = "✓" if hit else "✗"
        print(f"  {icon}  [{modality:6s}]  {query[:55]}")
        if retrieved_papers:
            print(f"           top-3: {retrieved_papers[:3]}")

        results_log.append({
            "query":             query,
            "modality":          modality,
            "expected_keywords": expected_keywords,
            "hit":               hit,
            "top_retrieved":     retrieved_papers[:3],
            "top_scores":        [round(s, 4) for s in retrieved_scores[:3]],
        })

    recall = hits / total if total else 0.0
    print(f"\nRecall@{k}: {hits}/{total} = {recall:.1%}")

    # Save results
    out_path = "evaluation/results.json"
    with open(out_path, "w") as f:
        json.dump({"recall_at_k": k, "recall": recall, "details": results_log}, f, indent=2)
    print(f"Results saved → {out_path}")

    return recall


def faithfulness_spot_check(k: int = 3) -> None:
    """
    Run a small set of queries through the full answer() pipeline
    and print the generated answers for manual inspection.
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
        result    = answer(q, retrieved)
        # Print first 400 chars of answer
        ans_preview = result["answer"].replace("\n", " ")[:400]
        print(f"A: {ans_preview}...")


if __name__ == "__main__":
    recall_at_k(k=5)
    faithfulness_spot_check(k=3)