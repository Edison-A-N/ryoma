from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional
from ryoma.core.llm import create_llm
from ryoma.core.logging import logger

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    provider: Optional[Literal["aws_bedrock", "azure_openai", "google_gemini"]] = None
    model_id: Optional[str] = None
    prompt: str


class ChatResponse(BaseModel):
    message: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        llm = create_llm(provider=request.provider, model_id=request.model_id)
        response = llm.chat(request.prompt)
        return ChatResponse(message=response)
    except Exception as e:
        logger.error(
            "Failed to generate response", error=str(e), provider=request.provider
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            llm = create_llm(provider=request.provider, model_id=request.model_id)
            for chunk in llm.stream_chat(request.prompt):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(
                "Failed to generate response", error=str(e), provider=request.provider
            )
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
