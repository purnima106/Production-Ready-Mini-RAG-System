import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(filepath="data/wiki_content.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return[item["text"] for item in data if item.get("text")]

def chunk_texts(texts, max_words=100):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) > 10:
                chunks.append(chunk)
    return chunks

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# 4. Store in FAISS
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 5. Save everything
def save_index(index, chunks, base_path="data/faiss_index"):
    faiss.write_index(index, f"{base_path}.index")
    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    texts = load_data()
    chunks = chunk_texts(texts)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))
    save_index(index, chunks)
    print(f"✅ Built FAISS index with {len(chunks)} chunks.")