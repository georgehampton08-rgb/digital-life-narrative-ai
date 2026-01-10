"""Usage Tracker — Local API Usage Tracking and Quota Awareness.

This module provides local tracking of Gemini API usage, adapted for the organizer package.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Pricing Constants (Gemini 2.0 Flash / 1.5 Pro)
# =============================================================================

PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash-exp": {
        "input": 0.0001,  # per 1K input tokens (estimated)
        "output": 0.0004, # per 1K output tokens (estimated)
    },
    "gemini-1.5-pro": {
        "input": 0.00125,  # per 1K input tokens
        "output": 0.00375,  # per 1K output tokens
    },
    "gemini-1.5-flash": {
        "input": 0.000075,
        "output": 0.0003,
    },
    "default": {
        "input": 0.0001,
        "output": 0.0004,
    },
}

@dataclass
class UsageRecord:
    """A single API call record."""
    id: str
    timestamp: datetime
    model: str
    operation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    success: bool
    error_type: str | None = None
    estimated_cost_usd: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        else:
            timestamp = ts or datetime.now(timezone.utc)
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=timestamp,
            model=data.get("model", "unknown"),
            operation=data.get("operation", "unknown"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            latency_ms=data.get("latency_ms", 0.0),
            success=data.get("success", True),
            error_type=data.get("error_type"),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "operation": self.operation,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_type": self.error_type,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

@dataclass
class UsageSummary:
    """Aggregated usage statistics."""
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    average_latency_ms: float = 0.0
    total_estimated_cost_usd: float = 0.0
    by_operation: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)

    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

class UsageTracker:
    """Local API usage tracking."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UsageTracker, cls).__new__(cls)
            return cls._instance

    def __init__(self, storage_path: Path | None = None) -> None:
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        if storage_path is None:
            from organizer.config import get_app_data_dir
            self._storage_path = get_app_data_dir() / "usage.json"
        else:
            self._storage_path = storage_path

        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._initialized = True
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def record(
        self,
        model: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool = True,
        error_type: str | None = None,
    ) -> UsageRecord:
        pricing = PRICING.get(model, PRICING["default"])
        estimated_cost = ((prompt_tokens / 1000) * pricing["input"]) + ((completion_tokens / 1000) * pricing["output"])

        record = UsageRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            model=model,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            estimated_cost_usd=estimated_cost,
        )

        with self._lock:
            self._records.append(record)
            self.save()

        return record

    def get_summary(self, start: datetime | None = None, end: datetime | None = None) -> UsageSummary:
        if start is None:
            start = datetime.min.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)

        with self._lock:
            records = [r for r in self._records if start <= r.timestamp < end]

        summary = UsageSummary(period_start=start, period_end=end)
        if not records:
            return summary

        summary.total_requests = len(records)
        summary.successful_requests = len([r for r in records if r.success])
        summary.failed_requests = len(records) - summary.successful_requests
        summary.total_prompt_tokens = sum(r.prompt_tokens for r in records)
        summary.total_completion_tokens = sum(r.completion_tokens for r in records)
        summary.total_tokens = sum(r.total_tokens for r in records)
        summary.total_estimated_cost_usd = sum(r.estimated_cost_usd for r in records)
        
        latencies = [r.latency_ms for r in records if r.latency_ms > 0]
        summary.average_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        for r in records:
            summary.by_operation[r.operation] = summary.by_operation.get(r.operation, 0) + 1
            summary.by_model[r.model] = summary.by_model.get(r.model, 0) + 1

        return summary

    def load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._records = [UsageRecord.from_dict(r) if hasattr(UsageRecord, 'from_dict') else UsageRecord(**r) for r in data]
        except Exception as e:
            logger.warning(f"Failed to load usage data: {e}")

    def save(self) -> None:
        try:
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._records], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save usage data: {e}")

def get_tracker() -> UsageTracker:
    return UsageTracker()
