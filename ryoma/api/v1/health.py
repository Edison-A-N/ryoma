from fastapi import APIRouter

from ryoma.core.config import settings
from ryoma.core.storage.vector import create_vector_database

router = APIRouter(tags=["health"])


def check_vector_db_health() -> tuple[bool, str]:
    try:
        db = create_vector_database(settings.STORAGE_VECTOR_TYPE)
        db.close()
        return True, "ok"
    except Exception as e:
        return False, f"error: {str(e)}"


@router.get("/healthz")
async def health_check():
    status = {"status": "ok", "services": {}}

    # Check vector database connection
    is_healthy, message = check_vector_db_health()
    status["services"]["vector_db"] = message
    if not is_healthy:
        status["status"] = "degraded"

    return status
