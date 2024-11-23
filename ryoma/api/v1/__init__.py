from fastapi import APIRouter
from .health import router as health_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(health_router)
