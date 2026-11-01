from datetime import datetime
from typing import List
from pydantic import BaseModel, Field, model_validator


class TelemetryStreamBatch(BaseModel):
    """
    Validation schema for an incoming window of 100 Hz industrial telemetry.
    Ensures all 5 physical streams are synchronized with identical sample lengths.
    """
    device_id: str = Field(..., description="Unique crucible induction furnace identifier")
    timestamp: datetime = Field(..., description="Timestamp of batch window capture")
    sampling_rate_hz: int = Field(default=100, ge=1, le=1000, description="Sampling frequency in Hz")

    # The 5 physical time-series arrays
    shell_temperature_Ts: List[float] = Field(..., description="Surface Shell Temperature array in Celsius")
    magnetic_flux_Bext: List[float] = Field(..., description="External Magnetic Flux Density array in mT")
    neutral_voltage_Vn: List[float] = Field(..., description="Neutral-to-Ground Potential array in Volts")
    leakage_current_In: List[float] = Field(..., description="Harmonic Leakage Current array in Amperes")
    coolant_pressure_delta_Pc: List[float] = Field(..., description="Cooling Water Differential Pressure in bar")

    @model_validator(mode="after")
    def verify_synchronized_stream_lengths(self) -> "TelemetryStreamBatch":
        """
        Validates that all 5 sensor arrays have identical lengths and meet
        the minimum sample window required for delay embedding (tau * (m - 1)).
        """
        lengths = {
            "Ts": len(self.shell_temperature_Ts),
            "Bext": len(self.magnetic_flux_Bext),
            "Vn": len(self.neutral_voltage_Vn),
            "In": len(self.leakage_current_In),
            "delta_Pc": len(self.coolant_pressure_delta_Pc),
        }

        first_len = lengths["Ts"]
        if first_len == 0:
            raise ValueError("Telemetry streams cannot be empty.")

        # Minimum window check for Takens embedding: (m - 1) * tau = (5 - 1) * 14 = 56 samples
        if first_len < 64:
            raise ValueError(
                f"Insufficient sample window size ({first_len}). "
                "Minimum required length for non-linear dynamics extraction is 64 samples."
            )

        # Check that all 5 sensor streams are strictly equal in length
        mismatched = {k: v for k, v in lengths.items() if v != first_len}
        if mismatched:
            raise ValueError(
                f"Sensor arrays are misaligned in length! "
                f"Expected all to match Ts ({first_len} samples), but got: {lengths}"
            )

        return self


class DiagnosticTaskResponse(BaseModel):
    """Immediate response payload returned by FastAPI after pushing to Redis."""
    task_id: str = Field(..., description="Celery task tracking UUID")
    device_id: str
    status: str = Field(default="PENDING", description="Current execution state")
    enqueued_at: datetime