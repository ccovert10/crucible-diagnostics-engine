import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CrucibleInferenceRecord(Base):
    __tablename__ = "crucible_inference_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(64), nullable=False, index=True)
    inference_task_id = Column(String(64), nullable=False, unique=True)
    signal_task_id = Column(String(64), nullable=True)
    predicted_failure_mode = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    lead_time_to_failure_hours = Column(Float, nullable=False)
    class_probabilities = Column(JSONB, nullable=False)
    sequence_steps_evaluated = Column(Integer, nullable=False)
    noise_reduction_stats = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_crucible_device_created", "device_id", "created_at"),
    )