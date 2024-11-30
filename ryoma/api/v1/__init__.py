from fastapi import APIRouter
from .chat import router as chat_router
from .health import router as health_router
from .html_reader import router as html_reader_router

v1_router = APIRouter()
v1_router.include_router(chat_router)
v1_router.include_router(health_router)
v1_router.include_router(html_reader_router)
