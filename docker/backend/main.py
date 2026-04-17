from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="LocalRAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    reply: str
    status: str

# In-memory store for demo
messages: list[dict] = []


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "backend"}


@app.post("/api/chat", response_model=MessageResponse)
def chat(request: MessageRequest):
    # Echo back for now — plug in RAG logic here
    reply = f"Backend received: '{request.message}'. Plug in your RAG pipeline here!"
    messages.append({"user": request.message, "bot": reply})
    return MessageResponse(reply=reply, status="success")


@app.get("/api/history")
def get_history():
    return {"history": messages}


@app.delete("/api/history")
def clear_history():
    messages.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
