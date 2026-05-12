from __future__ import annotations

from fastapi import APIRouter

from agentic_rec.models import InfoResponse
from agentic_rec.settings import settings

router = APIRouter()


@router.get("/info")
async def get_info() -> InfoResponse:
    """Return model configuration info."""
    return InfoResponse(
        embedder_name=settings.embedder_name,
        reranker_name=settings.reranker_name,
        llm_model=settings.llm_model,
    )
