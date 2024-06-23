from fastapi import FastAPI, HTTPException, Depends, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os
import time
import uuid
from functools import lru_cache
import uvicorn
import markdown
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API configuration

# Rate limiting configuration
RATE_LIMIT = 5  # requests per minute
RATE_WINDOW = 60  # seconds

# In-memory store for rate limiting
rate_limit_store = {}

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message]

# Rate limiting dependency
async def check_rate_limit(device_id: str = Cookie(None)):
    if not device_id:
        device_id = str(uuid.uuid4())

    current_time = time.time()
    if device_id in rate_limit_store:
        last_request_time, count = rate_limit_store[device_id]
        if current_time - last_request_time < RATE_WINDOW:
            if count >= RATE_LIMIT:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            rate_limit_store[device_id] = (last_request_time, count + 1)
        else:
            rate_limit_store[device_id] = (current_time, 1)
    else:
        rate_limit_store[device_id] = (current_time, 1)

    return device_id

# Document loading and retrieval
@lru_cache(maxsize=100)
def load_documents(directory: str = "local_files"):
    documents = []
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r') as file:
            documents.append(file.read())
    return documents

def retrieve_relevant_context(query: str, documents: List[str], top_n: int = 3):
    vectorizer = TfidfVectorizer().fit(documents + [query])
    doc_vectors = vectorizer.transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    relevant_contexts = [documents[i] for i in top_indices]
    return "\n".join(relevant_contexts)

# FastAPI routes
@app.post("/chat")
async def chat(
    conversation: Conversation,
    response: Response,
    device_id: str = Depends(check_rate_limit)
):
    documents = load_documents()
    context = retrieve_relevant_context(conversation.messages[-1].content, documents)

    augmented_messages = conversation.messages + [Message(role="system", content=f"Context: {context}")]

    chat_response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": m.role, "content": m.content} for m in augmented_messages])

    response.set_cookie(key="device_id", value=device_id, httponly=True, secure=True, samesite="strict")

    return {"response": chat_response.choices[0].message.content}

@app.post("/markdown_chat")
async def markdown_chat(
    conversation: Conversation,
    response: Response,
    device_id: str = Depends(check_rate_limit)
):
    documents = load_documents()
    context = retrieve_relevant_context(conversation.messages[-1].content, documents)

    augmented_messages = conversation.messages + [
        Message(role="system", content=f"Context: {context}"),
        Message(role="system", content="Please format your response in Markdown.")
    ]

    chat_response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": m.role, "content": m.content} for m in augmented_messages])

    markdown_response = chat_response.choices[0].message.content
    html_response = markdown.markdown(markdown_response)

    response.set_cookie(key="device_id", value=device_id, httponly=True, secure=True, samesite="strict")

    return {
        "markdown": markdown_response,
        "html": html_response
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)