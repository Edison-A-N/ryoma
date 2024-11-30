from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ryoma.core.workflow.html_reader import URLSummaryWorkflow, WorkflowResponse
from ryoma.core.logging import logger

router = APIRouter()


class HTMLReaderRequest(BaseModel):
    url: str


class HTMLReaderResponse(BaseModel):
    answer: WorkflowResponse


@router.post("/html-reader", response_model=HTMLReaderResponse)
async def read_html(request: HTMLReaderRequest):
    try:
        workflow = URLSummaryWorkflow()
        answer = await workflow.process(request.url)
        return HTMLReaderResponse(answer=answer)
    except Exception as e:
        logger.error(f"Failed to read HTML: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
