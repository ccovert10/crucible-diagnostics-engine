# Crucible Diagnostics Engine

A high-throughput, distributed diagnostic pipeline for metallurgical induction furnace crucibles. The engine processes high-frequency multi-channel telemetry streams, reconstructs phase-space dynamics via Takens' delay embedding and SVD filtering, and executes 1D Temporal Convolutional Network (1D-TCN) inference for sub-second failure mode classification and Remaining Useful Life (RUL) tracking.

---

## Architecture Overview

```text
[ Crucible Sensors (Ts, Bext, Vn, In, ΔPc) ]
                     │ (HTTP POST /telemetry)
                     ▼
          FastAPI Ingestion Gateway
                     │ (Celery Task)
                     ▼
             Redis Event Broker
                     │
                     ▼
           Celery Worker Pipeline
   ├── Signal Processing (Takens Embedding + SVD Denoising)
   ├── 1D-TCN Model Inference + Physical Energy Discrimination
   ├── Prometheus Latency & Probability Exposition
   └── PostgreSQL Persistence (crucible_inference_records)
         │                           │
         ▼                           ▼
 Prometheus / Grafana Dashboards    Historical Query REST API
```

### Core Diagnostic Modes
* **`CLASS_A_REFRACTORY_THINNING`**: Wall erosion characterized by thermal flux divergence and boundary magnetic field shift.
* **`CLASS_B_INDUCTION_COIL_MICRO_ARCING`**: High-frequency electrical discharges identified via neutral-point voltage ($V_n$) and coil current ($I_n$) localized phase variance.
* **`CLASS_C_COOLING_CIRCUIT_BLOCKAGE`**: Hydraulic choking detected by transient cooling circuit differential pressure ($\Delta P_c$) collapse.

---

## Technology Stack

* **Backend / Gateway:** FastAPI, Pydantic V2, Uvicorn
* **Task Queue & Broker:** Celery, Redis
* **Signal Processing & Deep Learning:** NumPy, SciPy (SVD Truncation), PyTorch (1D-TCN)
* **Storage & Persistence:** PostgreSQL 16, SQLAlchemy 2.0
* **Observability:** Prometheus, Grafana
* **Containerization & Testing:** Docker Compose, Pytest, HTTPX

---

## Quickstart

### 1. Prerequisites
* Docker Engine 24+ and Docker Compose v2+
* Python 3.11+ (for local test execution)

### 2. Environment Configuration
Create a `.env` file in the project root:

```env
PROJECT_NAME="Crucible Diagnostics Engine"
ENVIRONMENT="production"
API_V1_STR="/api/v1"

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=crucible_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/crucible_db

# Broker Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

### 3. Deploy the Stack
Spin up all infrastructure components (FastAPI, Redis, Celery Worker, PostgreSQL, Prometheus, Grafana):

```bash
docker compose up -d --build
```

Verify service status:

```bash
docker compose ps
```

---

## API Reference

### Health Check
```http
GET /health
```
Returns system runtime status, environment parameters, and Celery broker connection status.

### Historical Inference Query
```http
GET /api/v1/diagnostics/history
```
Query parameters:
* `device_id` (string, optional): Target crucible identifier (e.g., `CRUCIBLE-003`).
* `failure_mode` (string, optional): Target failure mode (e.g., `CLASS_C_COOLING_CIRCUIT_BLOCKAGE`).
* `min_confidence` (float, optional): Filter by minimum prediction probability (`0.0` - `1.0`).
* `limit` (int, default: 50): Number of records per page.
* `offset` (int, default: 0): Pagination offset.

**Example Request:**
```bash
curl.exe -s "http://localhost:8000/api/v1/diagnostics/history?device_id=CRUCIBLE-003&failure_mode=CLASS_C_COOLING_CIRCUIT_BLOCKAGE&limit=2"
```

Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

---

## Monitoring & Dashboards

* **Prometheus:** `http://localhost:9090` (Scrapes worker and gateway telemetry metrics).
* **Grafana:** `http://localhost:3000` (Pre-provisioned dashboard displaying real-time class distributions, SVD noise suppression ratios, and end-to-end task inference latency).

---

## Testing

Run the automated integration test suite against the live Docker stack:

```bash
pytest tests/test_history_api.py -v
```

---

## License

Proprietary / Industrial Internal Architecture.