"""Usage Tracker — Local API Usage Tracking and Quota Awareness.

This module provides local tracking of Gemini API usage, including:
- Token consumption per request
- Latency tracking
- Cost estimation
- Quota warnings
- Persistence across sessions

All data stays LOCAL on the user's machine. No external reporting.

Example:
    >>> from src.ai.usage_tracker import get_tracker, track_usage
    >>>
    >>> # Record a successful API call
    >>> record = track_usage(
    ...     model="gemini-1.5-pro",
    ...     operation="chapter_detection",
    ...     prompt_tokens=500,
    ...     completion_tokens=200,
    ...     latency_ms=1500.0,
    ... )
    >>>
    >>> # Check quota status
    >>> tracker = get_tracker()
    >>> status = tracker.check_quota()
    >>> if status.is_warning:
    ...     print(status.warning_message)
    >>>
    >>> # Get today's summary
    >>> summary = tracker.get_today_summary()
    >>> print(f"Total tokens today: {summary.total_tokens}")

Privacy:
- Only metadata is stored (tokens, timing, operation type)
- No prompt or response content is ever stored
- User owns all usage data
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
# Pricing Constants
# =============================================================================


PRICING: dict[str, dict[str, float]] = {
    "gemini-1.5-pro": {
        "input": 0.00125,   # per 1K input tokens
        "output": 0.00375,  # per 1K output tokens
    },
    "gemini-1.5-flash": {
        "input": 0.000075,
        "output": 0.0003,
    },
    "gemini-1.0-pro": {
        "input": 0.0005,
        "output": 0.0015,
    },
    "default": {
        "input": 0.001,
        "output": 0.003,
    },
}
"""Approximate pricing per 1K tokens.

Note: These are ESTIMATES only. Actual billing from Google may differ.
Pricing is subject to change. Check Google's official pricing page.
"""


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class UsageRecord:
    """A single API call record.
    
    Captures all relevant metadata about an API request for tracking
    and analysis purposes.
    
    Attributes:
        id: Unique record identifier.
        timestamp: When the call was made.
        model: Model used (e.g., "gemini-1.5-pro").
        operation: Operation type (e.g., "chapter_detection").
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        total_tokens: Sum of prompt + completion tokens.
        latency_ms: Request latency in milliseconds.
        success: Whether the call succeeded.
        error_type: Error type if failed.
        estimated_cost_usd: Estimated cost in USD.
    
    Example:
        >>> record = UsageRecord(
        ...     id="abc123",
        ...     timestamp=datetime.now(timezone.utc),
        ...     model="gemini-1.5-pro",
        ...     operation="narrative_generation",
        ...     prompt_tokens=500,
        ...     completion_tokens=200,
        ...     total_tokens=700,
        ...     latency_ms=1500.0,
        ...     success=True,
        ... )
    """
    
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
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation.
        """
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
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        """Create from dictionary.
        
        Args:
            data: Dictionary with record data.
        
        Returns:
            UsageRecord instance.
        """
        # Parse timestamp
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                timestamp = datetime.fromisoformat(ts)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.now(timezone.utc)
        
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


@dataclass
class UsageSummary:
    """Aggregated usage statistics.
    
    Provides a summary view of API usage over a time period.
    
    Attributes:
        period_start: Start of the summary period.
        period_end: End of the summary period.
        total_requests: Total number of requests.
        successful_requests: Count of successful requests.
        failed_requests: Count of failed requests.
        total_prompt_tokens: Sum of input tokens.
        total_completion_tokens: Sum of output tokens.
        total_tokens: Combined token count.
        average_latency_ms: Mean latency across requests.
        total_estimated_cost_usd: Estimated total cost.
        by_operation: Request counts by operation type.
        by_model: Request counts by model.
    
    Example:
        >>> summary = tracker.get_today_summary()
        >>> print(f"Success rate: {summary.success_rate():.1%}")
    """
    
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
        """Calculate success rate.
        
        Returns:
            Success rate as a float (0.0 to 1.0).
        """
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation.
        """
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "average_latency_ms": self.average_latency_ms,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "by_operation": self.by_operation,
            "by_model": self.by_model,
            "success_rate": self.success_rate(),
        }


@dataclass
class QuotaConfig:
    """Quota limits for warnings.
    
    Configure limits to receive warnings when approaching thresholds.
    Set limits to None to disable that quota check.
    
    Attributes:
        daily_token_limit: Maximum tokens per day.
        daily_request_limit: Maximum requests per day.
        monthly_cost_limit_usd: Maximum cost per month.
        warn_at_percentage: Percentage at which to warn (default 80%).
    
    Example:
        >>> config = QuotaConfig(
        ...     daily_token_limit=100000,
        ...     monthly_cost_limit_usd=10.0,
        ... )
    """
    
    daily_token_limit: int | None = None
    daily_request_limit: int | None = None
    monthly_cost_limit_usd: float | None = None
    warn_at_percentage: float = 80.0


@dataclass
class QuotaStatus:
    """Current quota status.
    
    Reports current usage against configured limits.
    
    Attributes:
        tokens_used_today: Tokens consumed today.
        tokens_remaining_today: Tokens remaining (if limit set).
        requests_today: Request count today.
        cost_this_month_usd: Estimated cost this month.
        is_warning: Whether any warning threshold reached.
        warning_message: Warning message if applicable.
        is_exceeded: Whether any limit is exceeded.
    
    Example:
        >>> status = tracker.check_quota()
        >>> if status.is_warning:
        ...     print(f"⚠️ {status.warning_message}")
    """
    
    tokens_used_today: int = 0
    tokens_remaining_today: int | None = None
    requests_today: int = 0
    cost_this_month_usd: float = 0.0
    is_warning: bool = False
    warning_message: str | None = None
    is_exceeded: bool = False


# =============================================================================
# Main Usage Tracker
# =============================================================================


class UsageTracker:
    """Local API usage tracking and quota awareness.
    
    Tracks all Gemini API calls locally for:
    - Usage monitoring
    - Cost estimation
    - Quota warnings
    - Historical analysis
    
    All data is stored locally in JSON format. No external reporting.
    
    Thread-safe for concurrent access.
    
    Attributes:
        _storage_path: Path to the storage file.
        _quota_config: Quota configuration.
        _records: In-memory usage records.
        _lock: Thread lock for concurrent access.
        _dirty: Whether records need saving.
    
    Example:
        >>> tracker = UsageTracker()
        >>> record = tracker.record(
        ...     model="gemini-1.5-pro",
        ...     operation="analysis",
        ...     prompt_tokens=500,
        ...     completion_tokens=200,
        ...     latency_ms=1500.0,
        ... )
        >>> print(f"Estimated cost: ${record.estimated_cost_usd:.4f}")
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        quota_config: QuotaConfig | None = None,
    ) -> None:
        """Initialize the usage tracker.
        
        Args:
            storage_path: Path to storage file. If None, uses default
                          location in config directory.
            quota_config: Quota configuration. Uses defaults if None.
        """
        # Determine storage path
        if storage_path is None:
            try:
                from src.config import get_config
                config = get_config()
                self._storage_path = config.paths.config_dir / "usage.json"
            except Exception:
                self._storage_path = Path.home() / ".dlna" / "usage.json"
        else:
            self._storage_path = storage_path
        
        self._quota_config = quota_config or QuotaConfig()
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._dirty = False
        
        # Ensure storage directory exists
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.load()
    
    # =========================================================================
    # Recording Methods
    # =========================================================================
    
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
        """Record a single API call.
        
        Thread-safe recording of API call metadata.
        
        Args:
            model: Model used (e.g., "gemini-1.5-pro").
            operation: Operation type (e.g., "chapter_detection").
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
            latency_ms: Request latency in milliseconds.
            success: Whether the call succeeded.
            error_type: Error type if failed.
        
        Returns:
            The created UsageRecord.
        
        Example:
            >>> record = tracker.record(
            ...     model="gemini-1.5-pro",
            ...     operation="narrative",
            ...     prompt_tokens=500,
            ...     completion_tokens=200,
            ...     latency_ms=1200.0,
            ... )
        """
        # Calculate estimated cost
        estimated_cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        
        # Create record
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
        
        # Add to records (thread-safe)
        with self._lock:
            self._records.append(record)
            self._dirty = True
        
        # Auto-save periodically
        if len(self._records) % 10 == 0:
            self._auto_save()
        
        # Log for debugging
        logger.debug(
            f"Recorded: {operation} on {model}, "
            f"{record.total_tokens} tokens, ${estimated_cost:.4f}"
        )
        
        return record
    
    def record_failure(
        self,
        model: str,
        operation: str,
        error_type: str,
        latency_ms: float = 0.0,
    ) -> UsageRecord:
        """Record a failed API call.
        
        Convenience method for recording failures with zero tokens.
        
        Args:
            model: Model that was targeted.
            operation: Operation that was attempted.
            error_type: Error type (e.g., "AIRateLimitError").
            latency_ms: Time before failure.
        
        Returns:
            The created failure record.
        
        Example:
            >>> record = tracker.record_failure(
            ...     model="gemini-1.5-pro",
            ...     operation="analysis",
            ...     error_type="AITimeoutError",
            ... )
        """
        return self.record(
            model=model,
            operation=operation,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error_type=error_type,
        )
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_today_summary(self) -> UsageSummary:
        """Get usage summary for today.
        
        Returns:
            UsageSummary for today (UTC).
        
        Example:
            >>> summary = tracker.get_today_summary()
            >>> print(f"Tokens: {summary.total_tokens}")
        """
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return self.get_summary(start, end)
    
    def get_summary(
        self,
        start: datetime,
        end: datetime,
    ) -> UsageSummary:
        """Get usage summary for a date range.
        
        Args:
            start: Start of period (inclusive).
            end: End of period (exclusive).
        
        Returns:
            UsageSummary for the period.
        
        Example:
            >>> from datetime import datetime, timedelta
            >>> start = datetime.now() - timedelta(days=7)
            >>> summary = tracker.get_summary(start, datetime.now())
        """
        with self._lock:
            records = [
                r for r in self._records
                if start <= r.timestamp < end
            ]
        
        if not records:
            return UsageSummary(period_start=start, period_end=end)
        
        # Aggregate statistics
        successful = [r for r in records if r.success]
        failed = [r for r in records if not r.success]
        
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.estimated_cost_usd for r in records)
        
        latencies = [r.latency_ms for r in records if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        # By operation
        by_operation: dict[str, int] = {}
        for r in records:
            by_operation[r.operation] = by_operation.get(r.operation, 0) + 1
        
        # By model
        by_model: dict[str, int] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + 1
        
        return UsageSummary(
            period_start=start,
            period_end=end,
            total_requests=len(records),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            average_latency_ms=avg_latency,
            total_estimated_cost_usd=total_cost,
            by_operation=by_operation,
            by_model=by_model,
        )
    
    def get_recent_records(self, count: int = 100) -> list[UsageRecord]:
        """Get most recent records.
        
        Args:
            count: Maximum number of records to return.
        
        Returns:
            List of most recent UsageRecord objects.
        """
        with self._lock:
            return list(reversed(self._records[-count:]))
    
    def get_records_for_operation(self, operation: str) -> list[UsageRecord]:
        """Get all records for a specific operation.
        
        Args:
            operation: Operation type to filter by.
        
        Returns:
            List of matching records.
        """
        with self._lock:
            return [r for r in self._records if r.operation == operation]
    
    # =========================================================================
    # Quota Methods
    # =========================================================================
    
    def check_quota(self) -> QuotaStatus:
        """Check current quota status.
        
        Evaluates usage against configured limits and returns
        status with warnings if thresholds are reached.
        
        Returns:
            QuotaStatus with current usage and warnings.
        
        Example:
            >>> status = tracker.check_quota()
            >>> if status.is_exceeded:
            ...     print("Quota exceeded!")
        """
        # Get today's summary
        today = self.get_today_summary()
        
        # Get this month's cost
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_summary = self.get_summary(month_start, now)
        
        tokens_today = today.total_tokens
        requests_today = today.total_requests
        cost_month = month_summary.total_estimated_cost_usd
        
        # Initialize status
        status = QuotaStatus(
            tokens_used_today=tokens_today,
            requests_today=requests_today,
            cost_this_month_usd=cost_month,
        )
        
        warnings = []
        exceeded = False
        
        # Check token limit
        if self._quota_config.daily_token_limit:
            limit = self._quota_config.daily_token_limit
            status.tokens_remaining_today = max(0, limit - tokens_today)
            
            usage_pct = (tokens_today / limit) * 100
            if usage_pct >= 100:
                exceeded = True
                warnings.append(f"Daily token limit exceeded ({tokens_today:,}/{limit:,})")
            elif usage_pct >= self._quota_config.warn_at_percentage:
                warnings.append(
                    f"Approaching daily token limit ({usage_pct:.0f}% used, "
                    f"{status.tokens_remaining_today:,} remaining)"
                )
        
        # Check request limit
        if self._quota_config.daily_request_limit:
            limit = self._quota_config.daily_request_limit
            usage_pct = (requests_today / limit) * 100
            
            if usage_pct >= 100:
                exceeded = True
                warnings.append(f"Daily request limit exceeded ({requests_today}/{limit})")
            elif usage_pct >= self._quota_config.warn_at_percentage:
                warnings.append(
                    f"Approaching daily request limit ({usage_pct:.0f}% used)"
                )
        
        # Check monthly cost limit
        if self._quota_config.monthly_cost_limit_usd:
            limit = self._quota_config.monthly_cost_limit_usd
            usage_pct = (cost_month / limit) * 100
            
            if usage_pct >= 100:
                exceeded = True
                warnings.append(f"Monthly cost limit exceeded (${cost_month:.2f}/${limit:.2f})")
            elif usage_pct >= self._quota_config.warn_at_percentage:
                warnings.append(
                    f"Approaching monthly cost limit ({usage_pct:.0f}% used, "
                    f"${cost_month:.2f}/${limit:.2f})"
                )
        
        status.is_exceeded = exceeded
        status.is_warning = len(warnings) > 0
        status.warning_message = "; ".join(warnings) if warnings else None
        
        return status
    
    def can_proceed(self, estimated_tokens: int = 0) -> tuple[bool, str | None]:
        """Check if we can make another request.
        
        Args:
            estimated_tokens: Estimated tokens for the request.
        
        Returns:
            Tuple of (can_proceed, reason_if_not).
        
        Example:
            >>> can, reason = tracker.can_proceed(estimated_tokens=1000)
            >>> if not can:
            ...     print(f"Cannot proceed: {reason}")
        """
        status = self.check_quota()
        
        if status.is_exceeded:
            return False, status.warning_message
        
        # Check if adding estimated tokens would exceed
        if self._quota_config.daily_token_limit and estimated_tokens > 0:
            if status.tokens_remaining_today is not None:
                if estimated_tokens > status.tokens_remaining_today:
                    return False, (
                        f"Request would exceed daily token limit "
                        f"({estimated_tokens:,} needed, "
                        f"{status.tokens_remaining_today:,} remaining)"
                    )
        
        return True, None
    
    def get_quota_warnings(self) -> list[str]:
        """Get list of active quota warnings.
        
        Returns:
            List of warning messages (empty if no warnings).
        """
        status = self.check_quota()
        if status.warning_message:
            return status.warning_message.split("; ")
        return []
    
    # =========================================================================
    # Cost Methods
    # =========================================================================
    
    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate cost for a request.
        
        Uses approximate pricing. Actual billing may differ.
        
        Args:
            model: Model name.
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
        
        Returns:
            Estimated cost in USD.
        
        Example:
            >>> cost = tracker.estimate_cost("gemini-1.5-pro", 1000, 500)
            >>> print(f"Estimated: ${cost:.4f}")
        """
        # Get pricing for model (use default if unknown)
        pricing = PRICING.get(model, PRICING["default"])
        
        # Calculate cost (price is per 1K tokens)
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_total_cost(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> float:
        """Get total estimated cost for a period.
        
        Args:
            start: Start of period (None = all time).
            end: End of period (None = now).
        
        Returns:
            Total estimated cost in USD.
        """
        if start is None:
            start = datetime.min.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)
        
        summary = self.get_summary(start, end)
        return summary.total_estimated_cost_usd
    
    # =========================================================================
    # Persistence Methods
    # =========================================================================
    
    def save(self) -> None:
        """Save records to storage file.
        
        Thread-safe save operation. Handles errors gracefully.
        """
        with self._lock:
            if not self._dirty:
                return
            
            records_data = [r.to_dict() for r in self._records]
        
        try:
            # Write to temp file first, then rename (atomic)
            temp_path = self._storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"records": records_data, "version": 1}, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self._storage_path)
            
            with self._lock:
                self._dirty = False
            
            logger.debug(f"Saved {len(records_data)} usage records")
            
        except Exception as e:
            logger.error(f"Failed to save usage records: {e}")
    
    def load(self) -> None:
        """Load records from storage file.
        
        Handles missing/corrupt files gracefully.
        """
        if not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            records_data = data.get("records", [])
            
            with self._lock:
                self._records = [
                    UsageRecord.from_dict(r) for r in records_data
                ]
                self._dirty = False
            
            logger.debug(f"Loaded {len(self._records)} usage records")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt usage file, starting fresh: {e}")
            with self._lock:
                self._records = []
        except Exception as e:
            logger.error(f"Failed to load usage records: {e}")
    
    def _auto_save(self) -> None:
        """Save if dirty (called periodically)."""
        if self._dirty:
            self.save()
    
    # =========================================================================
    # Cleanup Methods
    # =========================================================================
    
    def prune_old_records(self, days: int = 90) -> int:
        """Remove records older than specified days.
        
        Args:
            days: Age threshold in days.
        
        Returns:
            Count of records removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        with self._lock:
            original_count = len(self._records)
            self._records = [r for r in self._records if r.timestamp >= cutoff]
            removed = original_count - len(self._records)
            
            if removed > 0:
                self._dirty = True
        
        if removed > 0:
            logger.info(f"Pruned {removed} usage records older than {days} days")
            self.save()
        
        return removed
    
    def clear(self) -> None:
        """Clear all records.
        
        Use for testing or reset purposes.
        """
        with self._lock:
            self._records = []
            self._dirty = True
        self.save()
    
    # =========================================================================
    # Report Methods
    # =========================================================================
    
    def generate_usage_report(self) -> str:
        """Generate human-readable usage report.
        
        Returns:
            Formatted usage report string.
        
        Example:
            >>> print(tracker.generate_usage_report())
        """
        now = datetime.now(timezone.utc)
        
        # Time periods
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        today = self.get_summary(today_start, now)
        week = self.get_summary(week_start, now)
        month = self.get_summary(month_start, now)
        
        status = self.check_quota()
        
        lines = [
            "=" * 50,
            "📊 USAGE REPORT",
            "=" * 50,
            "",
            "TODAY:",
            f"  Requests: {today.total_requests}",
            f"  Tokens: {today.total_tokens:,}",
            f"  Est. Cost: ${today.total_estimated_cost_usd:.4f}",
            f"  Avg Latency: {today.average_latency_ms:.0f}ms",
            f"  Success Rate: {today.success_rate():.1%}",
            "",
            "THIS WEEK:",
            f"  Requests: {week.total_requests}",
            f"  Tokens: {week.total_tokens:,}",
            f"  Est. Cost: ${week.total_estimated_cost_usd:.4f}",
            "",
            "THIS MONTH:",
            f"  Requests: {month.total_requests}",
            f"  Tokens: {month.total_tokens:,}",
            f"  Est. Cost: ${month.total_estimated_cost_usd:.4f}",
            "",
        ]
        
        # Operations breakdown
        if month.by_operation:
            lines.append("BY OPERATION (this month):")
            for op, count in sorted(month.by_operation.items(), key=lambda x: -x[1]):
                lines.append(f"  {op}: {count}")
            lines.append("")
        
        # Quota status
        if status.is_warning or status.is_exceeded:
            lines.append("⚠️ QUOTA WARNINGS:")
            if status.warning_message:
                for warning in status.warning_message.split("; "):
                    lines.append(f"  • {warning}")
            lines.append("")
        
        lines.append("=" * 50)
        lines.append("Note: Costs are ESTIMATES only. Actual billing may differ.")
        
        return "\n".join(lines)
    
    def export_to_csv(self, path: Path) -> None:
        """Export records to CSV.
        
        Args:
            path: Path to write CSV file.
        
        Example:
            >>> tracker.export_to_csv(Path("usage.csv"))
        """
        import csv
        
        with self._lock:
            records = list(self._records)
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "timestamp", "model", "operation",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "latency_ms", "success", "error_type", "estimated_cost_usd"
            ])
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())
        
        logger.info(f"Exported {len(records)} records to {path}")


# =============================================================================
# Singleton and Convenience Functions
# =============================================================================


_tracker_instance: UsageTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> UsageTracker:
    """Get singleton tracker instance.
    
    Lazily initialized on first call.
    
    Returns:
        The global UsageTracker instance.
    
    Example:
        >>> tracker = get_tracker()
        >>> summary = tracker.get_today_summary()
    """
    global _tracker_instance
    
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = UsageTracker()
    
    return _tracker_instance


def track_usage(
    model: str,
    operation: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    success: bool = True,
    error_type: str | None = None,
) -> UsageRecord:
    """Convenience function to record usage.
    
    Uses the singleton tracker.
    
    Args:
        model: Model used.
        operation: Operation type.
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens.
        latency_ms: Latency in ms.
        success: Whether call succeeded.
        error_type: Error type if failed.
    
    Returns:
        The created UsageRecord.
    
    Example:
        >>> track_usage("gemini-1.5-pro", "chapter_detection", 500, 200, 1500.0)
    """
    return get_tracker().record(
        model=model,
        operation=operation,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        success=success,
        error_type=error_type,
    )
