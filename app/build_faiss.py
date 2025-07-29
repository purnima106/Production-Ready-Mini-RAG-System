import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def load_data(json_path="data/wiki_content.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert to LangChain Documents
    docs = [Document(page_content=item["text"], metadata={"source": item["source"]}) for item in data]
    return docs

def build_and_save_faiss():
    docs = load_data()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # Save index (.faiss) and docstore (.pkl)
    os.makedirs("data", exist_ok=True)
    vectorstore.save_local("data", index_name="faiss_index")  # ✅ This will save both files
    print("✅ FAISS index + .pkl saved in /data")

if __name__ == "__main__":
    build_and_save_faiss()
