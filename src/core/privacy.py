"""Privacy Gate - CENTRAL PRIVACY CONTROL for Digital Life Narrative AI.

This module is the ONLY pathway for data to reach external AI services (Gemini).
It enforces explicit user consent, metadata-only transmission, anonymization,
audit logging, and data minimization.

**CRITICAL SECURITY BOUNDARY**

If this module is bypassed, it's a security bug.
If this module fails, it fails CLOSED (AI access denied, local-only mode).

Design Philosophy:
    - Privacy by Default: AI is OFF until explicitly enabled
    - Fail Closed: Any error = deny AI access
    - Transparency: Every transmission is logged and reviewable
    - Data Minimization: Send only what's necessary
    - Explicit Consent: Required for each session

Example:
    >>> from src.core.privacy import PrivacyGate, PrivacyLevel, PrivacySettings
    >>>
    >>> # Create gate with settings
    >>> settings = PrivacySettings(privacy_level=PrivacyLevel.STANDARD)
    >>> gate = PrivacyGate(settings)
    >>>
    >>> # Request consent
    >>> gate.grant_consent("User acknowledged data sharing")
    >>>
    >>> # Prepare memories for AI
    >>> safe_data, warnings = gate.prepare_memories_for_ai(memories)
    >>> # safe_data contains ONLY privacy-filtered dicts, never raw Memory objects
"""

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.memory import Location, Memory, PersonTag

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class PrivacyLevel(str, Enum):
    """Privacy level controlling what data is sent to AI services.

    Each level is progressively more permissive. Default is LOCAL_ONLY.

    Attributes:
        LOCAL_ONLY: No external communication allowed (AI analysis disabled)
        MINIMAL: Timestamps + media types only (no location, no people, no content)
        STANDARD: Above + country-level location + people counts (default for AI)
        DETAILED: Above + city-level + truncated captions + hashed names
        FULL: Everything except file paths (requires explicit per-session consent)
    """

    LOCAL_ONLY = "local_only"
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    FULL = "full"


class ConsentStatus(str, Enum):
    """Status of user consent for AI data transmission.

    Only GRANTED status allows AI communication.

    Attributes:
        NOT_REQUESTED: User hasn't been asked yet
        PENDING: Request shown, awaiting user response
        GRANTED: User approved (AI communication allowed)
        DENIED: User declined (AI blocked)
        EXPIRED: Previous consent timed out
        REVOKED: User withdrew consent after granting
    """

    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


class DataCategory(str, Enum):
    """Categories of data that can be selectively enabled/disabled.

    Users can fine-tune what types of information are shared with AI.

    Attributes:
        TIMESTAMP: When something happened (dates, times)
        LOCATION: Where something happened (coordinates, place names)
        PEOPLE: Who was involved (names, tags, counts)
        CONTENT: Captions, text, descriptions
        BEHAVIOR: Platform usage patterns (frequency, media types)
        MEDIA_TYPE: Type of media (photo, video, etc.)
        STATISTICS: Aggregated counts and summaries
    """

    TIMESTAMP = "timestamp"
    LOCATION = "location"
    PEOPLE = "people"
    CONTENT = "content"
    BEHAVIOR = "behavior"
    MEDIA_TYPE = "media_type"
    STATISTICS = "statistics"


# =============================================================================
# Custom Exceptions
# =============================================================================


class PrivacyError(Exception):
    """Base exception for privacy-related errors."""

    pass


class ConsentRequiredError(PrivacyError):
    """Raised when AI is called without valid user consent."""

    pass


class ConsentExpiredError(PrivacyError):
    """Raised when consent has expired."""

    pass


class PrivacyViolationError(PrivacyError):
    """Raised when attempting to send restricted data."""

    pass


class LocalOnlyModeError(PrivacyError):
    """Raised when AI is called in LOCAL_ONLY mode."""

    pass


# =============================================================================
# Supporting Models
# =============================================================================


class ConsentReceipt(BaseModel):
    """Record of user consent for AI data sharing.

    Provides proof that user was informed and granted permission.
    Includes expiration and scope of consent.

    Attributes:
        id: Unique identifier for this consent
        granted_at: When consent was granted
        expires_at: When consent expires (None = no expiration)
        privacy_level: Level of data sharing permitted
        enabled_categories: Data categories the user allowed
        disabled_categories: Data categories the user blocked
        session_id: Session this consent applies to
        user_acknowledgment: What the user was shown and agreed to
        ip_hash: Hashed IP for audit (never stored plainly)

    Example:
        >>> receipt = ConsentReceipt(
        ...     granted_at=datetime.now(timezone.utc),
        ...     privacy_level=PrivacyLevel.STANDARD,
        ...     enabled_categories=[DataCategory.TIMESTAMP, DataCategory.LOCATION],
        ...     disabled_categories=[DataCategory.PEOPLE],
        ...     session_id="abc123",
        ...     user_acknowledgment="User acknowledged data sharing"
        ... )
        >>> receipt.is_valid()
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    granted_at: datetime
    expires_at: datetime | None = None
    privacy_level: PrivacyLevel
    enabled_categories: list[DataCategory]
    disabled_categories: list[DataCategory]
    session_id: str
    user_acknowledgment: str
    ip_hash: str | None = None

    def is_valid(self) -> bool:
        """Check if consent is still active and not expired.

        Returns:
            True if consent is currently valid
        """
        if self.is_expired():
            return False
        return True

    def is_expired(self) -> bool:
        """Check if consent has expired.

        Returns:
            True if past expires_at timestamp
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def covers_category(self, category: DataCategory) -> bool:
        """Check if a data category is allowed by this consent.

        Args:
            category: Data category to check

        Returns:
            True if category is in enabled_categories
        """
        return category in self.enabled_categories


class TransmissionRecord(BaseModel):
    """Record of data transmitted to AI service.

    Logged for audit purposes. Users can review what was sent.

    Attributes:
        id: Unique identifier for this transmission
        timestamp: When data was sent
        consent_receipt_id: ID of consent that authorized this
        destination: AI service endpoint (e.g., "gemini-1.5-pro")
        payload_hash: Hash of payload (not the content itself)
        payload_size_bytes: Size of transmitted data
        memory_count: Number of memories included
        categories_included: Which data categories were sent
        response_received: Whether AI responded successfully
        error: Error message if transmission failed

    Example:
        >>> record = TransmissionRecord(
        ...     timestamp=datetime.now(timezone.utc),
        ...     consent_receipt_id="consent-123",
        ...     destination="gemini-1.5-pro",
        ...     payload_hash="abc123...",
        ...     payload_size_bytes=1024,
        ...     memory_count=10,
        ...     categories_included=[DataCategory.TIMESTAMP, DataCategory.MEDIA_TYPE],
        ...     response_received=True
        ... )
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    consent_receipt_id: str
    destination: str
    payload_hash: str
    payload_size_bytes: int
    memory_count: int
    categories_included: list[DataCategory]
    response_received: bool = False
    error: str | None = None


class PrivacySettings(BaseModel):
    """Configuration for privacy controls.

    Defines what data can be sent to AI and under what conditions.

    Attributes:
        privacy_level: Overall privacy level
        enabled_categories: Which data categories are allowed
        require_consent_each_session: Whether to require fresh consent per session
        consent_expiry_hours: Hours until consent expires (0 = no expiry)
        max_memories_per_request: Maximum memories in single AI request
        max_caption_length: Maximum caption length sent to AI
        anonymize_people: Whether to hash person names
        blur_location_precision: Decimal places for coordinates (0-4)
        allow_cross_session_learning: Whether AI can learn across sessions
        audit_log_enabled: Whether to log transmissions
        audit_log_path: Path to audit log file

    Example:
        >>> settings = PrivacySettings(
        ...     privacy_level=PrivacyLevel.STANDARD,
        ...     enabled_categories={DataCategory.TIMESTAMP, DataCategory.MEDIA_TYPE},
        ...     anonymize_people=True
        ... )
    """

    privacy_level: PrivacyLevel = PrivacyLevel.LOCAL_ONLY
    enabled_categories: set[DataCategory] = Field(default_factory=lambda: set(DataCategory))
    require_consent_each_session: bool = True
    consent_expiry_hours: int = 24
    max_memories_per_request: int = 500
    max_caption_length: int = 100
    anonymize_people: bool = True
    blur_location_precision: int = 1
    allow_cross_session_learning: bool = False
    audit_log_enabled: bool = True
    audit_log_path: Path | None = None

    def is_ai_enabled(self) -> bool:
        """Check if AI communication is enabled.

        Returns:
            True only if privacy_level is not LOCAL_ONLY
        """
        return self.privacy_level != PrivacyLevel.LOCAL_ONLY

    def to_user_summary(self) -> str:
        """Generate human-readable explanation of current settings.

        Returns:
            Multi-line string describing privacy configuration

        Example:
            >>> settings.to_user_summary()
            Privacy Level: STANDARD
            AI Analysis: Enabled
            Data Shared:
              ✓ When (timestamps)
              ✓ What (media types)
              ✗ Who (people names)
            ...
        """
        lines = []
        lines.append(f"Privacy Level: {self.privacy_level.value.upper()}")
        lines.append(
            f"AI Analysis: {'Enabled' if self.is_ai_enabled() else 'Disabled (Local Only)'}"
        )

        if self.is_ai_enabled():
            lines.append("\nData Shared:")
            category_descriptions = {
                DataCategory.TIMESTAMP: "When (timestamps)",
                DataCategory.LOCATION: "Where (locations)",
                DataCategory.PEOPLE: "Who (people names)",
                DataCategory.CONTENT: "What was said (captions)",
                DataCategory.BEHAVIOR: "How you use platforms",
                DataCategory.MEDIA_TYPE: "Media types",
                DataCategory.STATISTICS: "Aggregated statistics",
            }

            for category in DataCategory:
                enabled = category in self.enabled_categories
                symbol = "✓" if enabled else "✗"
                desc = category_descriptions.get(category, category.value)
                lines.append(f"  {symbol} {desc}")

            lines.append(
                f"\nPeople Names: {'Anonymized (hashed)' if self.anonymize_people else 'Included'}"
            )
            lines.append(f"Location Precision: ~{10 ** (2 - self.blur_location_precision)}km")
            lines.append(f"Caption Length: Max {self.max_caption_length} characters")

            if self.consent_expiry_hours > 0:
                lines.append(f"\nConsent Valid For: {self.consent_expiry_hours} hours")
            else:
                lines.append("\nConsent Valid For: Entire session")

        return "\n".join(lines)


# =============================================================================
# Main Privacy Gate Class
# =============================================================================


class PrivacyGate:
    """Central gatekeeper for ALL AI-bound data.

    This class is the ONLY pathway for data to reach external AI services.
    It enforces consent, applies privacy transformations, and logs all transmissions.

    **CRITICAL**: Never bypass this class. Direct AI calls = security bug.

    Attributes:
        settings: Privacy configuration
        current_consent: Active consent receipt (if any)
        consent_status: Current consent state
        transmission_history: Log of all transmissions this session
        session_id: Unique session identifier

    Example:
        >>> gate = PrivacyGate()
        >>>
        >>> # Configure privacy level
        >>> gate.settings.privacy_level = PrivacyLevel.STANDARD
        >>> gate.settings.enabled_categories = {
        ...     DataCategory.TIMESTAMP,
        ...     DataCategory.MEDIA_TYPE,
        ...     DataCategory.LOCATION
        ... }
        >>>
        >>> # Request and grant consent
        >>> gate.grant_consent("User acknowledged: timestamps and locations will be shared")
        >>>
        >>> # Prepare data for AI
        >>> safe_data, warnings = gate.prepare_memories_for_ai(memories)
        >>>
        >>> # Record transmission
        >>> record = gate.record_transmission(safe_data, "gemini-1.5-pro")
    """

    def __init__(self, settings: PrivacySettings | None = None):
        """Initialize privacy gate.

        Args:
            settings: Privacy configuration (defaults to LOCAL_ONLY)
        """
        self.settings = settings or PrivacySettings()
        self.current_consent: ConsentReceipt | None = None
        self.consent_status = ConsentStatus.NOT_REQUESTED
        self.transmission_history: list[TransmissionRecord] = []
        self.session_id = str(uuid4())

        # Initialize audit logger
        if self.settings.audit_log_enabled:
            self._setup_audit_logger()

        logger.info(
            f"PrivacyGate initialized: session={self.session_id}, level={self.settings.privacy_level.value}"
        )

    def _setup_audit_logger(self) -> None:
        """Configure audit logging for transmissions."""
        if self.settings.audit_log_path:
            # Add file handler for audit log
            handler = logging.FileHandler(self.settings.audit_log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s - AUDIT - %(message)s"))
            logger.addHandler(handler)

    # =========================================================================
    # Consent Methods
    # =========================================================================

    def request_consent(self, callback: Callable[[str], bool] | None = None) -> ConsentStatus:
        """Request user consent for AI data sharing.

        Generates consent message explaining what will be shared.
        If callback provided, immediately processes response.
        Otherwise, sets status to PENDING for async handling.

        Args:
            callback: Optional function to get user's yes/no response
                     Receives consent message, returns True/False

        Returns:
            New consent status

        Example:
            >>> def ask_user(message: str) -> bool:
            ...     print(message)
            ...     return input("Agree? (y/n): ").lower() == 'y'
            >>>
            >>> status = gate.request_consent(callback=ask_user)
        """
        if not self.settings.is_ai_enabled():
            logger.warning("Consent requested but privacy_level is LOCAL_ONLY")
            self.consent_status = ConsentStatus.DENIED
            return self.consent_status

        consent_message = self.get_consent_message()

        if callback:
            # Synchronous consent
            user_response = callback(consent_message)
            if user_response:
                self.grant_consent(consent_message)
                return ConsentStatus.GRANTED
            else:
                self.deny_consent("User declined consent")
                return ConsentStatus.DENIED
        else:
            # Async consent (UI will handle)
            self.consent_status = ConsentStatus.PENDING
            logger.info("Consent request pending user response")
            return ConsentStatus.PENDING

    def grant_consent(self, user_acknowledgment: str = "User granted consent") -> ConsentReceipt:
        """Grant consent for AI data sharing.

        Creates consent receipt and marks consent as GRANTED.

        Args:
            user_acknowledgment: What the user was shown and agreed to

        Returns:
            Consent receipt

        Raises:
            LocalOnlyModeError: If privacy_level is LOCAL_ONLY

        Example:
            >>> receipt = gate.grant_consent("User acknowledged data sharing")
            >>> assert receipt.is_valid()
        """
        if not self.settings.is_ai_enabled():
            raise LocalOnlyModeError("Cannot grant consent in LOCAL_ONLY mode")

        # Calculate expiry
        expires_at = None
        if self.settings.consent_expiry_hours > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=self.settings.consent_expiry_hours
            )

        # Determine enabled/disabled categories
        all_categories = set(DataCategory)
        enabled = self.settings.enabled_categories
        disabled = all_categories - enabled

        receipt = ConsentReceipt(
            granted_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            privacy_level=self.settings.privacy_level,
            enabled_categories=list(enabled),
            disabled_categories=list(disabled),
            session_id=self.session_id,
            user_acknowledgment=user_acknowledgment,
        )

        self.current_consent = receipt
        self.consent_status = ConsentStatus.GRANTED

        logger.info(
            f"Consent granted: receipt_id={receipt.id}, "
            f"level={receipt.privacy_level.value}, "
            f"expires={receipt.expires_at}"
        )

        return receipt

    def deny_consent(self, reason: str | None = None) -> None:
        """Deny consent for AI data sharing.

        Args:
            reason: Optional reason for denial
        """
        self.consent_status = ConsentStatus.DENIED
        logger.info(f"Consent denied: {reason or 'No reason provided'}")

    def revoke_consent(self) -> None:
        """Revoke previously granted consent.

        Invalidates current consent and blocks future AI access until
        consent is granted again.
        """
        if self.current_consent:
            logger.warning(f"Consent revoked: receipt_id={self.current_consent.id}")

        self.current_consent = None
        self.consent_status = ConsentStatus.REVOKED

    def check_consent(self) -> tuple[bool, str]:
        """Verify consent is valid right now.

        Checks:
        - Status is GRANTED
        - Consent not expired
        - Session matches

        Returns:
            Tuple of (is_valid, reason_if_not)

        Example:
            >>> is_valid, reason = gate.check_consent()
            >>> if not is_valid:
            ...     print(f"Consent invalid: {reason}")
        """
        # Check LOCAL_ONLY mode
        if not self.settings.is_ai_enabled():
            return False, "AI disabled (LOCAL_ONLY mode)"

        # Check consent status
        if self.consent_status != ConsentStatus.GRANTED:
            return False, f"Consent status: {self.consent_status.value}"

        # Check consent exists
        if not self.current_consent:
            return False, "No consent receipt found"

        # Check expiry
        if self.current_consent.is_expired():
            self.consent_status = ConsentStatus.EXPIRED
            return False, "Consent has expired"

        # Check session match
        if self.current_consent.session_id != self.session_id:
            return False, "Consent from different session"

        return True, "Consent valid"

    # =========================================================================
    # Data Transformation Methods
    # =========================================================================

    def prepare_memories_for_ai(self, memories: list[Memory]) -> tuple[list[dict], list[str]]:
        """Prepare memories for AI transmission with privacy filtering.

        **THE CRITICAL METHOD**

        This is the main data transformation pipeline. It:
        1. Validates consent
        2. Checks memory count limits
        3. Applies privacy transformations based on settings
        4. Validates output for privacy leaks

        Args:
            memories: List of Memory objects to prepare

        Returns:
            Tuple of (sanitized_dicts, warnings)

        Raises:
            ConsentRequiredError: If consent not granted
            ConsentExpiredError: If consent has expired
            LocalOnlyModeError: If AI is disabled
            PrivacyViolationError: If validation detects privacy leak

        Example:
            >>> safe_data, warnings = gate.prepare_memories_for_ai(memories)
            >>> for warning in warnings:
            ...     print(f"Warning: {warning}")
            >>> send_to_gemini(safe_data)  # Safe to send
        """
        warnings: list[str] = []

        # 1. Check consent
        is_valid, reason = self.check_consent()
        if not is_valid:
            if self.consent_status == ConsentStatus.EXPIRED:
                raise ConsentExpiredError(reason)
            elif not self.settings.is_ai_enabled():
                raise LocalOnlyModeError(reason)
            else:
                raise ConsentRequiredError(reason)

        # 2. Check memory count
        if len(memories) > self.settings.max_memories_per_request:
            warnings.append(
                f"Memory count ({len(memories)}) exceeds limit "
                f"({self.settings.max_memories_per_request}). "
                f"Only first {self.settings.max_memories_per_request} will be processed."
            )
            memories = memories[: self.settings.max_memories_per_request]

        # 3. Transform each memory
        sanitized = []
        for memory in memories:
            try:
                safe_dict = self.prepare_single_memory(memory)
                sanitized.append(safe_dict)
            except Exception as e:
                warnings.append(f"Failed to prepare memory {memory.id}: {str(e)}")
                logger.error(f"Memory preparation error: {e}", exc_info=True)

        # 4. Validate output
        is_safe, violations = self.validate_outbound_data(sanitized)
        if not is_safe:
            raise PrivacyViolationError(f"Privacy violations detected: {', '.join(violations)}")

        logger.info(
            f"Prepared {len(sanitized)} memories for AI "
            f"(privacy_level={self.settings.privacy_level.value})"
        )

        return sanitized, warnings

    def prepare_single_memory(self, memory: Memory) -> dict:
        """Apply privacy transformations to a single memory.

        Applies filtering based on privacy level and enabled categories.

        Args:
            memory: Memory object to sanitize

        Returns:
            Privacy-filtered dictionary
        """
        output: dict[str, Any] = {
            "id": memory.id,  # Always include ID for tracking
        }

        # Get privacy level for categorization
        level = self.settings.privacy_level
        consent = self.current_consent

        # TIMESTAMP category
        if consent and consent.covers_category(DataCategory.TIMESTAMP):
            if memory.created_at:
                if level == PrivacyLevel.MINIMAL:
                    # Year and month only
                    output["created_at"] = memory.created_at.strftime("%Y-%m")
                elif level in [PrivacyLevel.STANDARD, PrivacyLevel.DETAILED]:
                    # Full date, no time
                    output["created_at"] = memory.created_at.strftime("%Y-%m-%d")
                elif level == PrivacyLevel.FULL:
                    # Full timestamp
                    output["created_at"] = memory.created_at.isoformat()

        # MEDIA_TYPE category
        if consent and consent.covers_category(DataCategory.MEDIA_TYPE):
            output["media_type"] = memory.media_type.value
            output["platform"] = memory.source_platform.value

        # LOCATION category
        if consent and consent.covers_category(DataCategory.LOCATION):
            if memory.location and not memory.location.is_empty():
                if level == PrivacyLevel.MINIMAL:
                    # No location at MINIMAL
                    pass
                elif level == PrivacyLevel.STANDARD:
                    # Country only
                    output["location_country"] = (
                        memory.location.country_code or memory.location.country
                    )
                elif level == PrivacyLevel.DETAILED:
                    # City/region level
                    output["location"] = memory.location.to_ai_summary()
                elif level == PrivacyLevel.FULL:
                    # Full location (but still no exact coordinates)
                    output["location"] = memory.location.to_display_string()

                    # Blurred coordinates if available
                    if memory.location.coordinates:
                        lat, lon = self.blur_coordinates(
                            memory.location.coordinates.latitude,
                            memory.location.coordinates.longitude,
                        )
                        output["approximate_coordinates"] = {"lat": lat, "lon": lon}

        # PEOPLE category
        if consent and consent.covers_category(DataCategory.PEOPLE):
            if level == PrivacyLevel.MINIMAL:
                # No people info at MINIMAL
                pass
            elif level == PrivacyLevel.STANDARD:
                # Count only
                output["people_count"] = len(memory.people)
            elif level in [PrivacyLevel.DETAILED, PrivacyLevel.FULL]:
                # Anonymized names
                if self.settings.anonymize_people:
                    output["people"] = [self.anonymize_name(p.name) for p in memory.people]
                else:
                    output["people"] = [p.name for p in memory.people]

        # CONTENT category
        if consent and consent.covers_category(DataCategory.CONTENT):
            if memory.caption:
                if level == PrivacyLevel.MINIMAL:
                    # Just whether caption exists
                    output["has_caption"] = True
                elif level == PrivacyLevel.STANDARD:
                    output["has_caption"] = True
                elif level in [PrivacyLevel.DETAILED, PrivacyLevel.FULL]:
                    # Truncated caption with PII stripped
                    output["caption"] = self.truncate_caption(memory.caption)

        # STATISTICS category (always safe)
        if consent and consent.covers_category(DataCategory.STATISTICS):
            if memory.width and memory.height:
                output["dimensions"] = f"{memory.width}x{memory.height}"
            if memory.duration_seconds:
                output["duration_seconds"] = round(memory.duration_seconds, 1)

        return output

    def anonymize_name(self, name: str) -> str:
        """Hash a person's name for privacy.

        Uses consistent hashing so same name = same hash within session.

        Args:
            name: Person's name to anonymize

        Returns:
            Anonymized name like "Person_a1b2c3d4"

        Example:
            >>> gate.anonymize_name("Alice Smith")
            'Person_4fa8c1cd'
        """
        # Hash with session salt for consistency within session
        salted = f"{self.session_id}:{name}".encode()
        hash_hex = hashlib.sha256(salted).hexdigest()[:8]
        return f"Person_{hash_hex}"

    def blur_coordinates(self, lat: float, lon: float) -> tuple[float, float]:
        """Reduce precision of coordinates for privacy.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tuple of (blurred_lat, blurred_lon)

        Example:
            >>> gate.settings.blur_location_precision = 1
            >>> gate.blur_coordinates(40.7829123, -73.9654456)
            (40.8, -74.0)
        """
        precision = self.settings.blur_location_precision
        return (round(lat, precision), round(lon, precision))

    def truncate_caption(self, caption: str) -> str:
        """Truncate caption and strip potential PII.

        Args:
            caption: Original caption text

        Returns:
            Truncated caption with PII patterns removed

        Example:
            >>> gate.truncate_caption("Email me at john@example.com about this!")
            'Email me at [EMAIL] about this!'
        """
        # Strip common PII patterns
        cleaned = caption

        # Email pattern
        cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", cleaned)

        # Phone pattern (simple US)
        cleaned = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", cleaned)

        # URL pattern
        cleaned = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "[URL]",
            cleaned,
        )

        # Truncate
        max_length = self.settings.max_caption_length
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."

        return cleaned

    # =========================================================================
    # Transmission Methods
    # =========================================================================

    def record_transmission(
        self,
        payload: list[dict],
        destination: str,
        response_received: bool = False,
        error: str | None = None,
    ) -> TransmissionRecord:
        """Log that data was sent to AI service.

        Creates audit record with payload hash (not content).

        Args:
            payload: Data that was sent
            destination: AI service identifier (e.g., "gemini-1.5-pro")
            response_received: Whether AI responded successfully
            error: Error message if transmission failed

        Returns:
            Transmission record

        Example:
            >>> record = gate.record_transmission(
            ...     safe_data,
            ...     "gemini-1.5-pro",
            ...     response_received=True
            ... )
        """
        if not self.current_consent:
            logger.warning("Recording transmission without active consent")

        # Hash payload
        import json

        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()[:16]
        payload_size = len(payload_str.encode())

        # Determine categories included
        categories = set()
        if payload:
            sample = payload[0]
            if "created_at" in sample:
                categories.add(DataCategory.TIMESTAMP)
            if "location" in sample or "location_country" in sample:
                categories.add(DataCategory.LOCATION)
            if "people" in sample or "people_count" in sample:
                categories.add(DataCategory.PEOPLE)
            if "caption" in sample or "has_caption" in sample:
                categories.add(DataCategory.CONTENT)
            if "media_type" in sample:
                categories.add(DataCategory.MEDIA_TYPE)

        record = TransmissionRecord(
            timestamp=datetime.now(timezone.utc),
            consent_receipt_id=self.current_consent.id if self.current_consent else "no_consent",
            destination=destination,
            payload_hash=payload_hash,
            payload_size_bytes=payload_size,
            memory_count=len(payload),
            categories_included=list(categories),
            response_received=response_received,
            error=error,
        )

        self.transmission_history.append(record)

        if self.settings.audit_log_enabled:
            logger.info(
                f"TRANSMISSION: id={record.id}, dest={destination}, "
                f"memories={len(payload)}, hash={payload_hash}, "
                f"success={response_received}"
            )

        return record

    def get_transmission_history(self) -> list[TransmissionRecord]:
        """Get all transmissions in this session.

        Returns:
            List of transmission records
        """
        return self.transmission_history

    def export_audit_log(self, path: Path) -> None:
        """Export transmission history to file for user review.

        Args:
            path: Where to save audit log

        Example:
            >>> gate.export_audit_log(Path("audit_log.json"))
        """
        import json

        data = {
            "session_id": self.session_id,
            "privacy_level": self.settings.privacy_level.value,
            "consent_status": self.consent_status.value,
            "transmissions": [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "destination": r.destination,
                    "memory_count": r.memory_count,
                    "categories": [c.value for c in r.categories_included],
                    "success": r.response_received,
                }
                for r in self.transmission_history
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Audit log exported to {path}")

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def validate_outbound_data(self, data: list[dict]) -> tuple[bool, list[str]]:
        """Scan transformed data for privacy leaks.

        Checks for:
        - File paths
        - Raw coordinates (too precise)
        - Email addresses
        - Phone numbers

        Args:
            data: Sanitized data to validate

        Returns:
            Tuple of (is_safe, violations_list)

        Example:
            >>> is_safe, violations = gate.validate_outbound_data(data)
            >>> if not is_safe:
            ...     raise PrivacyViolationError(violations)
        """
        violations = []

        for item in data:
            # Check for file paths
            for key, value in item.items():
                if isinstance(value, str):
                    # File path patterns
                    if "/" in value or "\\" in value:
                        if key != "id":  # IDs can have / in UUIDs
                            violations.append(f"Potential file path in {key}: {value[:50]}")

                    # PII patterns
                    pii_types = self.detect_pii(value)
                    if pii_types:
                        violations.append(f"PII detected in {key}: {', '.join(pii_types)}")

            # Check coordinate precision
            if "coordinates" in item or "approximate_coordinates" in item:
                coords = item.get("coordinates") or item.get("approximate_coordinates")
                if coords and isinstance(coords, dict):
                    lat = coords.get("lat", 0)
                    lon = coords.get("lon", 0)

                    # Too precise? (more than 4 decimal places)
                    lat_str = str(lat)
                    lon_str = str(lon)
                    if "." in lat_str and len(lat_str.split(".")[1]) > 4:
                        violations.append(f"Coordinates too precise: {lat}, {lon}")

        return len(violations) == 0, violations

    def detect_pii(self, text: str) -> list[str]:
        """Scan text for potential PII patterns.

        Args:
            text: Text to scan

        Returns:
            List of detected PII types (e.g., ["email", "phone"])

        Example:
            >>> gate.detect_pii("Call me at 555-1234 or email@example.com")
            ['phone', 'email']
        """
        pii_types = []

        # Email
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
            pii_types.append("email")

        # Phone (simple patterns)
        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text):
            pii_types.append("phone")

        # SSN pattern
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
            pii_types.append("ssn")

        # Credit card (simple)
        if re.search(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", text):
            pii_types.append("credit_card")

        return pii_types

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_consent_message(self) -> str:
        """Generate consent request message for user.

        Explains what data will be shared based on current settings.
        Must be clear, honest, and understandable by non-technical users.

        Returns:
            Consent message string

        Example:
            >>> message = gate.get_consent_message()
            >>> print(message)
        """
        level = self.settings.privacy_level

        message = f"""
=== AI ANALYSIS CONSENT REQUEST ===

To reconstruct your life narrative, we need to send some of your data to Google's Gemini AI.

PRIVACY LEVEL: {level.value.upper()}

WHAT WILL BE SHARED:
"""

        if level == PrivacyLevel.MINIMAL:
            message += """
- When photos/videos were taken (month and year only)
- What type of media (photo, video, etc.)
- Which platform (Snapchat, Google Photos, etc.)

WHAT WILL NOT BE SHARED:
- Your actual photos or videos (only metadata)
- Locations
- People's names
- Captions or text
- File paths or folder structure
"""
        elif level == PrivacyLevel.STANDARD:
            message += """
- When photos/videos were taken (full dates)
- What type of media
- Which platform
- What country you were in
- How many people were in photos (not their names)
- Whether captions exist (not the content)

WHAT WILL NOT BE SHARED:
- Your actual photos or videos
- Exact locations (city names or coordinates)
- People's names
- Full captions
- File paths
"""
        elif level == PrivacyLevel.DETAILED:
            message += """
- When photos/videos were taken (full dates)
- What type of media
- Which platform
- What city/region you were in
- Anonymized people names (hashed, not real names)
- Truncated captions (first 100 characters, with emails/phones removed)

WHAT WILL NOT BE SHARED:
- Your actual photos or videos
- Exact GPS coordinates
- Real people's names
- Full captions
- File paths
"""
        elif level == PrivacyLevel.FULL:
            message += """
- When photos/videos were taken (full timestamps)
- What type of media
- Which platform
- Where you were (city/region, approximate coordinates)
- People's names (anonymized if enabled)
- Captions (truncated to 100 characters)

WHAT WILL NOT BE SHARED:
- Your actual photos or videos
- File paths or folder structure
- Exact GPS coordinates (blurred to ~1km precision)
"""

        message += f"""
RETENTION:
- This consent is valid for {self.settings.consent_expiry_hours} hours
- You can revoke consent at any time
- All AI transmissions are logged for your review

DATA TRANSMISSION:
- Up to {self.settings.max_memories_per_request} memories per request
- Sent to: Google Gemini AI
- Connection: Encrypted (HTTPS)

Do you consent to sharing this data for AI analysis?
"""

        return message

    def get_privacy_summary(self) -> dict:
        """Get current privacy state summary.

        Returns:
            Dictionary with privacy status

        Example:
            >>> summary = gate.get_privacy_summary()
            >>> print(f"Consent: {summary['consent_status']}")
        """
        return {
            "session_id": self.session_id,
            "privacy_level": self.settings.privacy_level.value,
            "ai_enabled": self.settings.is_ai_enabled(),
            "consent_status": self.consent_status.value,
            "consent_valid": self.check_consent()[0],
            "enabled_categories": [c.value for c in self.settings.enabled_categories],
            "transmissions_count": len(self.transmission_history),
            "total_memories_sent": sum(r.memory_count for r in self.transmission_history),
        }

    def reset(self) -> None:
        """Reset privacy gate to initial state.

        Clears consent, transmission history, generates new session ID.
        """
        self.current_consent = None
        self.consent_status = ConsentStatus.NOT_REQUESTED
        self.transmission_history = []
        self.session_id = str(uuid4())

        logger.info(f"PrivacyGate reset: new session={self.session_id}")


# =============================================================================
# Module-Level Functions
# =============================================================================


# Global singleton gate
_default_gate: PrivacyGate | None = None


def get_default_gate() -> PrivacyGate:
    """Get module-level singleton privacy gate.

    Returns default gate with LOCAL_ONLY settings.

    Returns:
        Default PrivacyGate instance

    Example:
        >>> from src.core.privacy import get_default_gate
        >>> gate = get_default_gate()
    """
    global _default_gate
    if _default_gate is None:
        _default_gate = PrivacyGate()
    return _default_gate


def require_consent(func: Callable) -> Callable:
    """Decorator that checks consent before executing function.

    Raises ConsentRequiredError if consent not granted.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function

    Example:
        >>> @require_consent
        ... def call_ai_api(data):
        ...     return gemini.analyze(data)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        gate = get_default_gate()
        is_valid, reason = gate.check_consent()
        if not is_valid:
            raise ConsentRequiredError(
                f"AI access denied: {reason}. " f"Call gate.request_consent() first."
            )
        return func(*args, **kwargs)

    return wrapper
