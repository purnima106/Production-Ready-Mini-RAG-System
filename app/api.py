from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot.chatbot import get_chatbot
import uvicorn

app = FastAPI()
qa_chain = get_chatbot()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_api(req: QueryRequest):
    try:
        result = qa_chain.invoke(req.query)  
        return {
            "response": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
