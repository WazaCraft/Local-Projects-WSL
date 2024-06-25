from fastapi import FastAPI, HTTPException, Depends, Cookie, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
import time
import uuid
from functools import lru_cache
import uvicorn
import markdown
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq()

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

@app.post("/voice")
async def voice(
    response: Response,
    file: UploadFile = File(...),
    device_id: str = Depends(check_rate_limit)
):
    # Transcribe the audio file using Groq Whisper API
    file_content = await file.read()
    transcription = groq_client.audio.transcriptions.create(
        file=(file.filename, file_content),
        model="whisper-large-v3",
        response_format="json",
        language="en",
        temperature=0.0
    )

    transcribed_text = transcription.text  # Access the text attribute directly
    
    # Process the transcribed text with the chat endpoint
    documents = load_documents()
    context = retrieve_relevant_context(transcribed_text, documents)

    conversation = Conversation(messages=[
        Message(role="user", content=transcribed_text),
        Message(role="system", content=f"Context: {context}")
    ])

    chat_response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": m.role, "content": m.content} for m in conversation.messages])

    chat_text = chat_response.choices[0].message.content

    # Convert the chat text to speech using the TTS API
    speech_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=chat_text
    )

    audio_file_path = "/tmp/speech.mp3"
    speech_response.stream_to_file(audio_file_path)

    response.set_cookie(key="device_id", value=device_id, httponly=True, secure=True, samesite="strict")

    return FileResponse(audio_file_path, media_type="audio/mpeg", filename="response.mp3")

# Run the application
if __name__ == "__main__":
    uvicorn.run("alt:app", host="0.0.0.0", port=8000, reload=True)
