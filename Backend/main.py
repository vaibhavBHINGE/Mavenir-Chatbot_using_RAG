from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Import the agent builder from your agents.py file
from agents import build_agent
# Initialize FastAPI app
app = FastAPI(title="Mavenir 3GPP RAG Agent API")

print("Starting API Server and building Agent...")
# We build the agent once when the server starts, not on every request
try:
    agent_with_history = build_agent()
    print("Agent successfully loaded and ready to receive queries.")
except Exception as e:
    print(f"CRITICAL ERROR loading agent: {e}")
    agent_with_history = None

# Define the expected JSON payload format
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not agent_with_history:
        raise HTTPException(status_code=500, detail="Agent failed to initialize. Check server logs.")
        
    try:
        # LangChain requires the session_id to be passed inside the 'config' dictionary
        response = agent_with_history.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # The agent's final response is stored in the 'output' key
        return {"reply": response["output"]}
        
    except Exception as e:
        # Catch any LLM or tool parsing errors gracefully
        return {"reply": f"Agent encountered an internal error: {str(e)}. Please try asking in a different way."}

# Standard health check endpoint
@app.get("/")
async def root():
    return {"status": "3GPP Agent API is running."}