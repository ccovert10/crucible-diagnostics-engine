from fastapi import APIRouter
from app.api.v1.endpoints import diagnostics

api_router = APIRouter()
api_router.include_router(diagnostics.router, prefix="/diagnostics", tags=["Industrial Diagnostics"])