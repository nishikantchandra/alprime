"""Root FastAPI router aggregator."""

from fastapi import APIRouter

from app.api.endpoints import documents, health, query

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
