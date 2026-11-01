from celery import Celery
from app.config import get_settings

settings = get_settings()


def create_celery() -> Celery:
    """
    Factory function: Instantiates and configures the Celery application.
    Enforces late acknowledgments, single prefetch limits, and automated task discovery.
    """
    celery_instance = Celery(
        "crucible_worker",
        broker=settings.REDIS_URL,
        backend=settings.REDIS_URL,
    )

    celery_instance.conf.update(
        # Task Serialization (JSON only prevents remote code execution vulnerabilities)
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,

        # Result Backend Expiry
        result_expires=settings.CELERY_TASK_RESULT_EXPIRES,

        # Concurrency & Workload Distribution Tuning
        worker_prefetch_multiplier=1,      # Fair distribution: prevents idle workers
        task_acks_late=True,               # Acknowledge ONLY after task returns successfully
        task_reject_on_worker_lost=True,   # Re-queue task if worker process is OOM-killed

        # State Tracking
        task_track_started=True,           # Persist 'STARTED' state to Redis
    )

    # Automatic task module discovery across domain packages
    celery_instance.autodiscover_tasks([
        "app.tasks.signal_processing",
        "app.tasks.inference",
    ])

    return celery_instance


# Instantiated client referenced by CLI: celery -A app.celery_app:celery_app worker
celery_app = create_celery()