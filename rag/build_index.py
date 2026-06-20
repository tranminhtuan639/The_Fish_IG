# rag/build_index.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ─── Config ───────────────────────────────────────────────
KB_DIR      = "data/knowledge_base"
CHROMA_DIR  = "rag/chroma_db"
COLLECTION  = "fish_knowledge"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE  = 300

# ─── Chunk văn bản ────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE):
    words  = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ─── Main ─────────────────────────────────────────────────
def build_index():
    print("📚 Đang build RAG index...\n")

    embedder = SentenceTransformer(EMBED_MODEL)
    client   = chromadb.PersistentClient(path=CHROMA_DIR)

    # Xóa collection cũ nếu có
    try:
        client.delete_collection(COLLECTION)
        print("🗑️  Đã xóa collection cũ\n")
    except:
        pass

    collection = client.get_or_create_collection(COLLECTION)

    all_ids, all_docs, all_metas, all_embeds = [], [], [], []

    # Đọc cấu trúc thư mục mới:
    # knowledge_base/
    # ├── arowana/
    # │   ├── variants.md
    # │   ├── care.md
    # │   └── food.md
    # ├── oscar/
    # └── ...

    for species in os.listdir(KB_DIR):
        species_dir = os.path.join(KB_DIR, species)

        # Bỏ qua nếu không phải folder
        if not os.path.isdir(species_dir):
            continue

        print(f"📂 {species}:")

        for fname in os.listdir(species_dir):
            if not fname.endswith(".md"):
                continue

            topic  = fname.replace(".md", "")   # variants / care / food
            fpath  = os.path.join(species_dir, fname)

            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            print(f"   {fname}: {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                doc_id = f"{species}_{topic}_{i}"
                embed  = embedder.encode(chunk).tolist()

                all_ids.append(doc_id)
                all_docs.append(chunk)
                all_metas.append({
                    "species": species,   # "arowana"
                    "topic":   topic,     # "variants" / "care" / "food"
                    "chunk_id": i,
                })
                all_embeds.append(embed)

        print()

    collection.add(
        ids        = all_ids,
        documents  = all_docs,
        metadatas  = all_metas,
        embeddings = all_embeds,
    )

    print(f"✅ Done! Đã index {len(all_ids)} chunks vào ChromaDB")
    print(f"   Thư mục: {CHROMA_DIR}")

if __name__ == "__main__":
    build_index()