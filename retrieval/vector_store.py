"""
Index embeddings into Qdrant (local, on-disk).
Handles page images, text pages, and figure/table crops.
"""
import json
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

EMBED_FILE = "data/embeddings.pt"
META_FILE  = "data/page_metadata.json"
TEXT_META  = "data/text_metadata.json"
FIG_META   = "data/figure_metadata.json"
COLLECTION = "econ_papers"
VECTOR_DIM = 128


def build_index():
    # creates a local vector database
    client = QdrantClient(path="data/qdrant_db")
# If an old index exists → delete it .... to start fresh every time 
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        # Creates a new index with the dim and distance cosine bec it measure the similarite between two embeding  
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    embeddings = torch.load(EMBED_FILE, weights_only=False)
# Load metadata and convert to maps to  look on the page id imnstade searsh on all the list 
    with open(META_FILE) as f:
        page_metadata = json.load(f)
    page_map = {m["page_id"]: m for m in page_metadata}

    try:
        with open(TEXT_META, encoding="utf-8") as f:
            text_metadata = json.load(f)
    except FileNotFoundError:
        text_metadata = []
    text_map = {m["text_id"]: m for m in text_metadata}

    try:
        with open(FIG_META) as f:
            fig_metadata = json.load(f)
    except FileNotFoundError:
        fig_metadata = []
    fig_map = {m["fig_id"]: m for m in fig_metadata}

    points = []
    # loopthrough all the embed 
    for idx, (item_id, emb) in enumerate(embeddings.items()):
        # get the first vector for indexing from the vector that have multible vector 
        first_vec = emb[0].numpy().tolist()

        if item_id in page_map:
            meta    = page_map[item_id]
            payload = {
                "item_id":      item_id,
                "modality":     "image",
                "content_type": "page",
                "paper_id":     meta.get("paper_id", ""),
                "page_number":  meta.get("page_number", 0),
                "image_path":   meta.get("image_path", ""),
                "caption":      "",
            }

        elif item_id in fig_map:
            meta    = fig_map[item_id]
            payload = {
                "item_id":      item_id,
                "modality":     "image",
                "content_type": meta.get("type", "figure"),
                "paper_id":     meta.get("paper_id", ""),
                "page_number":  meta.get("page_number", 0),
                "image_path":   meta.get("image_path", ""),
                "caption":      meta.get("caption", ""),
            }

        else:
            meta    = text_map.get(item_id, {})
            payload = {
                "item_id":      item_id,
                "modality":     "text",
                "content_type": "text",
                "paper_id":     meta.get("paper_id", ""),
                "page_number":  meta.get("page_number", 0),
                "image_path":   "",
                "caption":      "",
            }

        # Single append at the end, to creat the qdrant point 
        points.append(PointStruct(id=idx, vector=first_vec, payload=payload))
# insert all the vector to can searsh on it 
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} items into '{COLLECTION}'")
    return client


if __name__ == "__main__":
    build_index()