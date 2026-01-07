import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from pydantic import BaseModel
import uvicorn
from advancedrag import RAGChatbot
import logging

# Set logging levels
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pyngrok").setLevel(logging.ERROR)

# Models for API requests
class QuestionRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None

class ChatDeleteRequest(BaseModel):
    chat_id: str

# Create FastAPI application
app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = RAGChatbot()
chatbot_initialized = False

@app.post("/initialize")
async def initialize_chatbot(background_tasks: BackgroundTasks):
    """Initialize the chatbot"""
    global chatbot_initialized

    if chatbot_initialized:
        return {"status": "success", "message": "Chatbot already initialized"}

    background_tasks.add_task(initialize_chatbot_task)
    return {"status": "initializing", "message": "Initialization started in background"}

async def initialize_chatbot_task():
    """Background task for initialization"""
    global chatbot_initialized

    try:
        success = chatbot.initialize()
        chatbot_initialized = success
        print(f"Chatbot initialization: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        chatbot_initialized = False

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    """Process a question and return an answer"""
    global chatbot_initialized

    # Check if chatbot is initialized
    if not chatbot_initialized:
        try:
            chatbot_initialized = chatbot.initialize()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

    # Process question
    response = chatbot.ask_question(request.question)

    # Get or create chat ID
    chat_id = request.chat_id if request.chat_id else str(uuid.uuid4())

    # Get or create title
    if not request.chat_id:
        title = chatbot.suggest_title(request.question, response.get("result", ""))
    else:
        existing_chat = chatbot.get_chat_by_id(chat_id)
        title = existing_chat["title"] if existing_chat else "Chat"

    # Update messages
    messages = []
    existing_chat = chatbot.get_chat_by_id(chat_id)
    if existing_chat:
        messages = existing_chat.get("messages", [])

    # Add new messages
    messages.append({"role": "user", "content": request.question})
    messages.append({"role": "assistant", "content": response.get("result", "")})

    # Update chat history
    chatbot.update_chat_history(chat_id, title, messages)

    # Format source documents
    source_documents = []
    if 'source_documents' in response:
        source_documents = [
            {"content": doc.page_content[:200] + "...", "source": doc.metadata.get("source", "Unknown")}
            for doc in response['source_documents']
        ]

    return {
        "chat_id": chat_id,
        "title": title,
        "question": request.question,
        "response": response.get("result", ""),
        "source_documents": source_documents
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing"""
    # Check if file type is supported
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in chatbot.supported_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    # Create upload directory if it doesn't exist
    if not os.path.exists(chatbot.upload_dir):
        os.makedirs(chatbot.upload_dir)

    # Save the file
    file_path = os.path.join(chatbot.upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the file
    documents = chatbot.process_document(file_path)

    return {
        "filename": file.filename,
        "status": "success" if documents else "error",
        "message": f"File uploaded and processed successfully" if documents else "File uploaded but processing failed"
    }

@app.get("/documents")
async def list_documents():
    """List all available documents"""
    try:
        if not os.path.exists(chatbot.upload_dir):
            return {"documents": []}

        documents = [
            f for f in os.listdir(chatbot.upload_dir)
            if os.path.isfile(os.path.join(chatbot.upload_dir, f))
            and os.path.splitext(f)[1].lower() in chatbot.supported_extensions
        ]

        return {"documents": documents}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/chats")
async def get_chats():
    """Get all chats"""
    chats = chatbot.get_chat_history()
    return {"chats": chats}

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    """Get a specific chat by ID"""
    chat = chatbot.get_chat_by_id(chat_id)

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    return chat

@app.delete("/chat")
async def delete_chat(request: ChatDeleteRequest):
    """Delete a chat by ID"""
    success = chatbot.delete_chat(request.chat_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete chat")

    return {"status": "success", "message": "Chat deleted successfully"}

@app.post("/reset")
async def reset_chatbot():
    """Reset the chatbot state"""
    global chatbot_initialized

    success = chatbot.reset()
    chatbot_initialized = False

    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset chatbot")

    return {"status": "success", "message": "Chatbot reset successfully"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0",
        "endpoints": {
            "initialize": "/initialize",
            "ask_question": "/ask_question",
            "upload": "/upload",
            "documents": "/documents",
            "chats": "/chats",
            "reset": "/reset"
        }
    }

if __name__ == "__main__":
    # Start ngrok tunnel
    try:
        public_url = ngrok.connect(8000)
        print(f"\n✅ FastAPI app is live at: {public_url}")
        print(f"📍 Local URL: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs\n")
    except Exception as e:
        print(f"⚠️ Ngrok connection failed: {str(e)}")
        print("Running without ngrok tunnel. Access at http://localhost:8000\n")
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)