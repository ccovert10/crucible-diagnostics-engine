import os
import sys
import time
from datetime import datetime, timezone
import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import httpx


def generate_targeted_telemetry(mode: str, n_samples: int = 100) -> dict:
    t = np.linspace(0, 1.0, n_samples)

    if mode == "A":  # Refractory Thinning
        Ts = 1350.0 + 35.0 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 2.0, n_samples)
        Bext = 1.45 + 0.15 * np.cos(2 * np.pi * 1.0 * t) + np.random.normal(0, 0.02, n_samples)
        Vn = 12.0 + np.random.normal(0, 0.5, n_samples)
        In = 1.5 + np.random.normal(0, 0.1, n_samples)
        delta_Pc = 4.2 + np.random.normal(0, 0.05, n_samples)
        device_id = "CRUCIBLE-001"

    elif mode == "B":  # Coil Micro-Arcing (High-frequency neutral voltage/current spikes)
        Ts = 1150.0 + np.random.normal(0, 2.0, n_samples)
        Bext = 1.1 + np.random.normal(0, 0.05, n_samples)
        # Random high-voltage micro-arc bursts
        micro_arcs = np.random.choice([0.0, 1.0], size=n_samples, p=[0.85, 0.15]) * np.random.uniform(25.0, 60.0, n_samples)
        Vn = 12.0 + micro_arcs + np.random.normal(0, 1.5, n_samples)
        In = 1.5 + (micro_arcs * 0.18) + np.random.normal(0, 0.2, n_samples)
        delta_Pc = 4.2 + np.random.normal(0, 0.05, n_samples)
        device_id = "CRUCIBLE-002"

    elif mode == "C":  # Sudden Cooling Circuit Blockage / Flow Stagnation
        # Abrupt step drop in coolant delta Pc at sample index 40
        delta_Pc = np.ones(n_samples) * 4.2
        delta_Pc[40:] = 0.35 + np.random.normal(0, 0.02, n_samples - 40)

        # Immediate thermal runaway following pressure loss
        Ts = np.ones(n_samples) * 1150.0
        Ts[40:] = 1150.0 + np.cumsum(np.random.uniform(4.0, 9.0, n_samples - 40))

        # Nominal electrical behavior
        Bext = 1.1 + np.random.normal(0, 0.02, n_samples)
        Vn = 12.0 + np.random.normal(0, 0.2, n_samples)
        In = 1.5 + np.random.normal(0, 0.05, n_samples)
        device_id = "CRUCIBLE-003"
        
    return {
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sampling_rate_hz": 100.0,
        "shell_temperature_Ts": [round(float(x), 3) for x in Ts],
        "magnetic_flux_Bext": [round(float(x), 4) for x in Bext],
        "neutral_voltage_Vn": [round(float(x), 3) for x in Vn],
        "leakage_current_In": [round(float(x), 3) for x in In],
        "coolant_pressure_delta_Pc": [round(float(x), 3) for x in delta_Pc],
    }


def run_multiclass_stress_stream():
    client = httpx.Client(base_url="http://localhost:8000", timeout=10.0)
    modes = ["A", "B", "C"] * 6  # 18 batches total across all 3 modes

    print(f"Streaming {len(modes)} targeted batches across Classes A, B, and C...")

    for i, mode in enumerate(modes, start=1):
        payload = generate_targeted_telemetry(mode=mode)
        res = client.post("/api/v1/diagnostics/process-telemetry", json=payload)
        
        if res.status_code == 202:
            task_id = res.json().get("task_id", "")[:8]
            print(f"[{i:02d}/18] Mode {mode} enqueued -> HTTP {res.status_code} | Task: {task_id}...")
        else:
            print(f"[{i:02d}/18] Mode {mode} failed -> HTTP {res.status_code}: {res.text}")
            
        time.sleep(0.25)

    print("\nTelemetry stream finished. Check the Failure Mode Distribution panel in Grafana.")


if __name__ == "__main__":
    run_multiclass_stress_stream()