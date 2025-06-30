from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from llama_index.core.memory import Memory
from agent import agent
import os
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio and load environment variables
nest_asyncio.apply()
load_dotenv()

app = FastAPI(
    title="IT Support Agent API",
    description="A LlamaIndex-powered IT support agent API",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (in production, use Redis or database)
sessions: Dict[str, Dict] = {}

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionCreate(BaseModel):
    session_name: Optional[str] = "New Session"

class SessionResponse(BaseModel):
    session_id: str
    session_name: str
    message_count: int

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "IT Support Agent API is running", "status": "healthy"}

@app.post("/sessions", response_model=SessionResponse)
async def create_session(session_data: SessionCreate):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "name": session_data.session_name,
        "messages": [
            {"role": "assistant", "content": "👋 Hello! How can I help you with TechCorp IT today?"}
        ],
        "memory": Memory.from_defaults(token_limit=30000)
    }
    
    return SessionResponse(
        session_id=session_id,
        session_name=session_data.session_name,
        message_count=1
    )

@app.get("/sessions", response_model=List[SessionResponse])
async def list_sessions():
    """List all active sessions"""
    return [
        SessionResponse(
            session_id=sid,
            session_name=session_data["name"],
            message_count=len(session_data["messages"])
        )
        for sid, session_data in sessions.items()
    ]

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages from a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"messages": sessions[session_id]["messages"]}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_message: ChatMessage):
    """Send a message to the IT support agent"""
    try:
        # Handle session management
        if chat_message.session_id and chat_message.session_id in sessions:
            session_id = chat_message.session_id
            session = sessions[session_id]
        else:
            # Create new session if none provided or session doesn't exist
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "name": "Auto-created Session",
                "messages": [
                    {"role": "assistant", "content": "👋 Hello! How can I help you with TechCorp IT today?"}
                ],
                "memory": Memory.from_defaults(token_limit=30000)
            }
            session = sessions[session_id]
        
        # Add user message to session
        session["messages"].append({"role": "user", "content": chat_message.message})
        
        # Set agent memory to session memory
        agent.memory = session["memory"]
        
        # Get response from agent
        response = agent.chat(chat_message.message)
        agent_response = response.response if hasattr(response, "response") else str(response)
        
        # Add agent response to session
        session["messages"].append({"role": "assistant", "content": agent_response})
        
        return ChatResponse(response=agent_response, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "agent_status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



