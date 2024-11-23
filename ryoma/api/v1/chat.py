from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from ryoma.core.llm import create_llm
from ryoma.core.logging import logger

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    provider: Literal["aws_bedrock", "azure_openai", "google_gemini"]
    model_id: str
    prompt: str


class ChatResponse(BaseModel):
    message: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(
            "Processing chat request", provider=request.provider, model=request.model_id
        )
        llm = create_llm(provider=request.provider, model_id=request.model_id)
        response = llm.chat(request.prompt)
        return ChatResponse(message=response)
    except Exception as e:
        logger.error(
            "Failed to generate response", error=str(e), provider=request.provider
        )
        raise HTTPException(status_code=500, detail=str(e))
