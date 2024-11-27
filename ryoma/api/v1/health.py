from fastapi import APIRouter
from pymilvus import connections

from ryoma.core.config import settings

router = APIRouter(tags=["health"])


def check_milvus_health() -> tuple[bool, str]:
    try:
        connections.connect(
            alias="default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT
        )
        connections.disconnect("default")
        return True, "ok"
    except Exception as e:
        return False, f"error: {str(e)}"


@router.get("/healthz")
async def health_check():
    status = {"status": "ok", "services": {}}

    # Check Milvus connection
    is_healthy, message = check_milvus_health()
    status["services"]["milvus"] = message
    if not is_healthy:
        status["status"] = "degraded"

    return status
