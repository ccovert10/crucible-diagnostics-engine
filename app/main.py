from typing import Optional
from fastapi import Depends, FastAPI, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.api.v1.router import api_router
from app.celery_app import celery_app  # Ensures Celery runtime bindings
from app.db.models import CrucibleInferenceRecord
from app.db.session import SessionLocal
from app.schemas.history import InferenceHistoryResponse, InferenceRecordOut

settings = get_settings()


def get_db():
    """Request-scoped database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_application() -> FastAPI:
    """
    FastAPI App Factory:
    Builds and configures the web application with metadata, CORS, and API routers.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Cross-Origin Resource Sharing (CORS) for industrial dashboards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API v1 routes
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/health", tags=["System Health"])
    def health_check():
        return {
            "status": "HEALTHY",
            "project": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
            "broker_configured": bool(celery_app.conf.broker_url),
        }

    @app.get(
        f"{settings.API_V1_STR}/diagnostics/history",
        response_model=InferenceHistoryResponse,
        status_code=status.HTTP_200_OK,
        tags=["Diagnostics"],
    )
    def get_inference_history(
        device_id: Optional[str] = Query(
            None,
            description="Filter by specific crucible device ID (e.g. CRUCIBLE-001)",
        ),
        failure_mode: Optional[str] = Query(
            None,
            description="Filter by predicted failure mode (e.g. CLASS_A_REFRACTORY_THINNING)",
        ),
        min_confidence: Optional[float] = Query(
            None,
            ge=0.0,
            le=1.0,
            description="Minimum prediction confidence threshold",
        ),
        limit: int = Query(50, ge=1, le=500, description="Page limit"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        db: Session = Depends(get_db),
    ):
        query = select(CrucibleInferenceRecord)

        if device_id:
            query = query.where(CrucibleInferenceRecord.device_id == device_id)
        if failure_mode:
            query = query.where(
                CrucibleInferenceRecord.predicted_failure_mode == failure_mode
            )
        if min_confidence is not None:
            query = query.where(CrucibleInferenceRecord.confidence >= min_confidence)

        # Count total matching records
        count_stmt = select(func.count()).select_from(query.subquery())
        total_count = db.scalar(count_stmt) or 0

        # Paginate and order by latest creation timestamp
        stmt = (
            query.order_by(desc(CrucibleInferenceRecord.created_at))
            .offset(offset)
            .limit(limit)
        )
        results = db.scalars(stmt).all()

        return InferenceHistoryResponse(
            total_records=total_count,
            limit=limit,
            offset=offset,
            records=[InferenceRecordOut.model_validate(row) for row in results],
        )

    return app


app = create_application()