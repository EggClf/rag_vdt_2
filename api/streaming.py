from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time
from typing import Generator, Optional, List, Dict, Any

import sys 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import dotenv
dotenv.load_dotenv(os.path.join(current_dir, '..', '.env'))


from src.chat import Chat

# Global variable to hold the chat agent
chat_agent = None


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "default-model"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = True

# Your streaming chat function
def chat_stream(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """
    Streaming chat function that yields content chunks.
    """
    for chunk in chat_agent.stream(messages, model):
        yield chunk

def create_app(
    model_name: str = "qwen/qwen2.5-7b-instruct",
    collection_name: str = "demo_multihop",
    qdrant_host: str = os.getenv("QDRANT_HOST", "http://localhost:6333"),
    device: str = 'cuda'
) -> FastAPI:
    """
    Factory function to create FastAPI app with configurable parameters.
    
    Args:
        model_name: The model name to use for the chat agent
        collection_name: Qdrant collection name
        qdrant_host: Qdrant host URL
        device: Device to use for inference ('cuda' or 'cpu')
    
    Returns:
        Configured FastAPI application
    """
    global chat_agent
    
    # Initialize the FastAPI app
    app = FastAPI(
        title="Multihop Backend API",
        description="Streaming chat API with RAG capabilities",
        version="1.0.0"
    )
    
    # Initialize the chat agent with provided parameters
    chat_agent = Chat(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        device=device,
        model_name=model_name
    )
    
    @app.get("/")
    def root():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": model_name,
            "collection": collection_name,
            "device": device,
            "qdrant_host": qdrant_host
        }
    
    @app.get("/health")
    def health():
        """Detailed health check."""
        return {
            "status": "healthy",
            "chat_agent_initialized": chat_agent is not None,
            "config": {
                "model_name": model_name,
                "collection_name": collection_name,
                "qdrant_host": qdrant_host,
                "device": device
            }
        }

    @app.post("/v1/chat/completions")
    def stream_chat(request: ChatRequest):
        """
        Streaming chat endpoint that mimics OpenAI's API format.
        """
        def event_generator():
            # Convert Pydantic models to dictionaries
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
            # Initial response with metadata
            yield f"""data: {json.dumps({
                'id': 'chatcmpl-1',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': request.model,
                'choices': [{
                    'delta': {'role': 'assistant'},
                    'index': 0,
                    'finish_reason': None
                }]
            })}\n\n"""
            
            # Stream the content chunks
            chunk_id = 2
            for content in chat_stream(
                messages,
                request.model,
                request.temperature,
                request.max_tokens
            ):
                yield f"""data: {json.dumps({
                'id': f'chatcmpl-{chunk_id}',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': request.model,
                'choices': [{
                    'delta': {'content': content},
                    'index': 0,
                    'finish_reason': None
                }]
            })}\n\n"""
                chunk_id += 1
            
            # Final chunk with finish_reason
            yield f"""data: {json.dumps({
                'id': f'chatcmpl-{chunk_id}',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': request.model,
                'choices': [{
                    'delta': {},
                    'index': 0,
                    'finish_reason': 'stop'
                }]
            })}\n\n"""
            
            # End of stream marker
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    return app


# Default app instance for backward compatibility
# app = create_app()

if __name__ == "__main__":
    app = create_app()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8910)