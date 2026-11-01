from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class InferenceRecordOut(BaseModel):
    id: UUID
    device_id: str
    inference_task_id: str
    signal_task_id: Optional[str] = None
    predicted_failure_mode: str
    confidence: float
    lead_time_to_failure_hours: float
    class_probabilities: Dict[str, float]
    sequence_steps_evaluated: int
    noise_reduction_stats: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InferenceHistoryResponse(BaseModel):
    total_records: int
    limit: int
    offset: int
    records: List[InferenceRecordOut]