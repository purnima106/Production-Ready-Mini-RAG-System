import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load FAISS vectorstore (must already be created)
import os

def build_vectorstore(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Check if .pkl exists before trying to load
    faiss_path = "data/faiss_index.pkl"
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"❌ Missing {faiss_path}. Run build_faiss.py to create it.")

    # Proceed to load safely
    vectorstore = FAISS.load_local(
        folder_path="data",
        embeddings=embeddings,
        index_name="faiss_index",
        allow_dangerous_deserialization=True
    )
    return vectorstore


# Load small local LLM (like t5-small)
def load_local_llm():
    model_id = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=pipe)

# Build the QA chain
def get_chatbot():
    vectorstore = build_vectorstore()  # ✅ must return FAISS object, NOT dict
    local_llm = load_local_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=3),
        return_source_documents=True
    )
    return qa_chain
