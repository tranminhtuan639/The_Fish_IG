# rag/retriever.py
import chromadb
from sentence_transformers import SentenceTransformer

# ─── Config ───────────────────────────────────────────────
CHROMA_DIR  = "rag/chroma_db"
COLLECTION  = "fish_knowledge"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Load 1 lần duy nhất
_embedder   = None
_collection = None

def _get_collection():
    global _embedder, _collection
    if _collection is None:
        _embedder   = SentenceTransformer(EMBED_MODEL)
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION)
    return _embedder, _collection

# ─── Retrieve theo topic cụ thể ───────────────────────────
def retrieve_by_topic(species: str, topic: str) -> str:
    """
    Lấy toàn bộ nội dung 1 topic của 1 loài.
    topic: "variants" | "care" | "food"

    Dùng khi cần lấy đúng 1 mục — ví dụ:
    - Lấy variants để đoán biến thể
    - Lấy care / food để trả lời câu hỏi cụ thể
    """
    _, collection = _get_collection()

    results = collection.get(
        where = {"$and": [
            {"species": species},
            {"topic":   topic},
        ]}
    )

    docs = results["documents"]
    if not docs:
        return ""

    return "\n\n".join(docs)

# ─── Retrieve theo câu hỏi tự do ──────────────────────────
def retrieve_by_query(species: str, query: str, top_k: int = 3) -> str:
    """
    Tìm kiếm semantic theo câu hỏi user trong phạm vi 1 loài.

    Dùng khi user hỏi tự do — ví dụ:
    - "cá rồng ăn gì?"
    - "nhiệt độ bể bao nhiêu?"
    """
    embedder, collection = _get_collection()

    query_embed = embedder.encode(f"{species} {query}").tolist()

    results = collection.query(
        query_embeddings = [query_embed],
        n_results        = top_k,
        where            = {"species": species},
    )

    docs = results["documents"][0]
    if not docs:
        return ""

    return "\n\n".join(docs)

# ─── Retrieve toàn bộ loài (cho lần đầu giới thiệu) ──────
def retrieve_all(species: str) -> dict:
    """
    Lấy toàn bộ variants + care + food của 1 loài.
    Trả về dict để prompt_builder ghép linh hoạt.
    """
    return {
        "variants": retrieve_by_topic(species, "variants"),
        "care":     retrieve_by_topic(species, "care"),
        "food":     retrieve_by_topic(species, "food"),
    }


# ─── Test nhanh ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Test retrieve_by_topic (variants - arowana):")
    print("=" * 50)
    print(retrieve_by_topic("arowana", "variants"))

    print("\n" + "=" * 50)
    print("Test retrieve_by_query (arowana ăn gì):")
    print("=" * 50)
    print(retrieve_by_query("arowana", "cá rồng ăn gì"))

    print("\n" + "=" * 50)
    print("Test retrieve_all (snakehead):")
    print("=" * 50)
    data = retrieve_all("snakehead")
    for topic, content in data.items():
        print(f"\n[{topic}]")
        print(content[:200], "...")