"""
Retrieve top-K items using MaxSim (ColBERT late interaction).
"""
import torch
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor

EMBED_FILE = "data/embeddings.pt"
COLLECTION = "econ_papers"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

_model, _processor, _embeddings, _client = None, None, None, None


def _load():
    global _model, _processor, _embeddings, _client
    if _model is None:
        print("Loading ColPali model for retrieval ...")
        _model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE).eval()
        _processor  = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
        _embeddings = torch.load(EMBED_FILE, weights_only=False)
        _client     = QdrantClient(path="data/qdrant_db")
        print(f"Loaded {len(_embeddings)} embeddings on {DEVICE}")


def maxsim_score(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
    scores = torch.einsum("qd,pd->qp",
                          query_emb.float(),
                          doc_emb.float())
    return scores.max(dim=1).values.sum().item()


def retrieve(query: str, top_k: int = 5) -> list:
    _load()

    with torch.no_grad():
        inputs = _processor.process_queries([query]).to(DEVICE)
        q_emb  = _model(**inputs)[0].cpu()

    mean_q = q_emb.mean(dim=0).numpy().tolist()
    hits   = _client.query_points(
        collection_name=COLLECTION,
        query=mean_q,
        limit=top_k * 4,
    ).points

    scored = []
    for hit in hits:
        item_id = hit.payload.get("item_id")
        if item_id not in _embeddings:
            continue
        d_emb = _embeddings[item_id]
        score = maxsim_score(q_emb, d_emb)
        hit.payload["_score"] = score
        scored.append((score, hit.payload))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    test_query = "What is the effect of monetary policy on inflation?"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query, top_k=3)
    for rank, (score, meta) in enumerate(results, 1):
        print(f"  #{rank}  score={score:.3f}  "
              f"item={meta.get('item_id')}  "
              f"type={meta.get('content_type')}  "
              f"paper={meta.get('paper_id')}  "
              f"page={meta.get('page_number')}")