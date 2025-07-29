# 🛫 AI-Powered Chatbot for Jewel Changi Airport (RAG-based)

This project is an AI-powered **question-answering chatbot** built using the **Retrieval-Augmented Generation (RAG)** technique. It answers user queries related to Jewel Changi Airport and the Changi Skytrain system — all without using the OpenAI API.

---

## 🚀 Features

- 🔍 Web-scraped Wikipedia content using `BeautifulSoup`
- 🧠 Local LLM (`Flan-T5`) used for lightweight response generation
- 📚 FAISS vectorstore built with HuggingFace embeddings
- 🧾 RAG pipeline for semantic search + generation
- ⚡ FastAPI backend with `/query` endpoint for easy testing
- 🧪 Postman-compatible for manual query testing

---

### 🗂️ Project Structure

```
AI_Airport_Chatbot/
├── app/
│   ├── api.py                # FastAPI backend
│   ├── build_faiss.py        # Builds FAISS index from JSON
│   └── chatbot/
│       └── chatbot.py        # RAG chain construction logic
├── data/
│   ├── wiki_content.json     # Scraped wiki data
│   ├── faiss_index.faiss     # FAISS vector index
│   └── faiss_index.pkl       # Serialized document store
├── requirements.txt
└── README.md
```
yaml
Copy
Edit

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/AI_Airport_Chatbot.git
cd AI_Airport_Chatbot

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate     # For Windows

# 3. Install dependencies
pip install -r requirements.txt
📥 Step-by-Step Guide
🧾 1. Scrape Wikipedia and save as JSON (already done)
If needed, re-scrape using your custom script or update wiki_content.json.

📚 2. Build FAISS Index
bash
Copy
Edit
python app/build_faiss.py
This generates:

data/faiss_index.faiss

data/faiss_index.pkl

⚙️ 3. Run the FastAPI Server
bash
Copy
Edit
uvicorn app.api:app --reload
📬 How to Query via Postman
POST URL: http://localhost:8000/query

Headers: Content-Type: application/json

Body:

json
Copy
Edit
{
  "query": "What is the Rain Vortex at Jewel Changi Airport?"
}
🧠 Tech Stack
Area	Tools Used
Language Model	flan-t5-base (local via HuggingFace)
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Store	FAISS
Backend	FastAPI, Pydantic, Uvicorn
Data Source	Wikipedia (via BeautifulSoup)

🔮 Future Improvements
Switch to OpenAI (if API key is available) for better answers

Add frontend UI (Streamlit or React)

Highlight source document snippets

Add authentication to the API

Deploy on Hugging Face Spaces or Render

🧑‍💻 Author
Purnima Nahata
LinkedIn | GitHub

