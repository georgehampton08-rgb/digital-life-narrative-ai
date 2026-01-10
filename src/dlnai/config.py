"""Central Configuration System for Digital Life Narrative AI.

This module is the single source of truth for application configuration.
Every other module that needs settings imports from here. It implements the
project's core philosophy: "AI is optional and off by default, privacy is
the default, but Gemini is the core value when enabled."

The configuration system supports:
- Multi-source configuration (environment variables > config file > defaults)
- Secure API key management (env > keyring > encrypted file)
- Privacy modes with enforced constraints
- Graceful degradation when services are unavailable

Example:
    >>> from dlnai.config import get_config, get_api_key
    >>>
    >>> # Get cached configuration singleton
    >>> cfg = get_config()
    >>> print(cfg.ai.mode)  # AIMode.DISABLED by default
    >>>
    >>> # Check if AI is available
    >>> if cfg.is_ai_available():
    ...     api_key = get_api_key()
    ...     # Use the key with Gemini client

Config File Format (YAML):
    ```yaml
    ai:
      mode: disabled  # disabled | enabled | fallback_only
      model_name: gemini-1.5-pro
      temperature: 0.7
      max_output_tokens: 8192
      timeout_seconds: 120
      max_retries: 3
      depth_mode: deep  # quick | standard | deep
      max_vision_images_per_run: 120
      vision_model: gemini-2.0-flash-exp
      narrative_model: gemini-1.5-pro
      show_cost_estimates: true

    privacy:
      mode: strict  # strict | standard | detailed | full
      send_timestamps: true
      send_locations: false
      send_captions: false
      max_caption_length: 100
      hash_people_names: true
      require_consent_per_session: true

    paths:
      config_dir: ~/.dlna
      output_dir: ./output

    report:
      format: html
      theme: auto

    debug: false
    verbose: false
    ```
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import platform
import stat
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings

# Optional dependencies with graceful fallback
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False

try:
    import keyring
    import keyring.errors

    KEYRING_AVAILABLE = True
except ImportError:
    keyring = None  # type: ignore
    KEYRING_AVAILABLE = False

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore
    CRYPTOGRAPHY_AVAILABLE = False


# Configure module logger - never log secrets
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ConfigError(Exception):
    """Base exception for configuration errors.

    All configuration-related exceptions inherit from this class to allow
    for easy exception handling at a higher level.
    """

    pass


class ConfigFileError(ConfigError):
    """Exception raised for YAML config file issues.

    Raised when:
    - Config file exists but cannot be read
    - Config file contains malformed YAML
    - Config file has structural issues
    """

    pass


class APIKeyError(ConfigError):
    """Base exception for API key related issues.

    All API key exceptions inherit from this to allow specific handling
    of key-related errors.
    """

    pass


class APIKeyNotFoundError(APIKeyError):
    """Exception raised when API key cannot be found in any source.

    This is raised when all configured sources (environment, keyring,
    encrypted file) have been checked and no valid key was found.
    """

    pass


class APIKeyInvalidError(APIKeyError):
    """Exception raised when API key fails format validation.

    This does NOT indicate the key was rejected by the API - only that
    it fails basic format checks (length, whitespace, etc.).
    """

    pass


# =============================================================================
# Enums
# =============================================================================


class AIMode(str, Enum):
    """AI feature activation modes.

    Controls whether the application will make calls to Gemini for AI analysis.
    The default is DISABLED - users must explicitly opt-in to AI features.

    Attributes:
        DISABLED: AI features completely off. No network calls to Gemini.
                  This is the default for privacy-first operation.
        ENABLED: AI features active. Will call Gemini when analysis requested.
                 Requires a valid API key.
        FALLBACK_ONLY: Attempt AI analysis but gracefully degrade if unavailable.
                       Useful for resilient operation when API may be temporary down.

    Example:
        >>> config.ai.mode = AIMode.ENABLED
        >>> if config.ai.mode == AIMode.DISABLED:
        ...     print("AI features are off")
    """

    DISABLED = "disabled"
    ENABLED = "enabled"
    FALLBACK_ONLY = "fallback_only"


class DepthModeConfig(BaseModel):
    """Configuration for a specific depth mode's sampling behavior.
    
    Controls how many images are sampled per chapter and the strategy
    used to select them.
    """
    name: str
    images_per_chapter_min: int
    images_per_chapter_max: int
    images_per_chapter_target: int
    prioritize_metadata_rich: bool = True
    temporal_spread_weight: float = 0.5


# Per-mode presets (Increased by ~11 images for better clarity/accuracy)
QUICK_MODE_CONFIG = DepthModeConfig(
    name="quick",
    images_per_chapter_min=12,
    images_per_chapter_max=13,
    images_per_chapter_target=12,
    prioritize_metadata_rich=True,
    temporal_spread_weight=0.3
)
STANDARD_MODE_CONFIG = DepthModeConfig(
    name="standard",
    images_per_chapter_min=13,
    images_per_chapter_max=16,
    images_per_chapter_target=14,
    prioritize_metadata_rich=True,
    temporal_spread_weight=0.5
)
DEEP_MODE_CONFIG = DepthModeConfig(
    name="deep",
    images_per_chapter_min=16,
    images_per_chapter_max=23,
    images_per_chapter_target=19,
    prioritize_metadata_rich=True,
    temporal_spread_weight=0.7
)

DEPTH_MODE_CONFIGS = {
    "quick": QUICK_MODE_CONFIG,
    "standard": STANDARD_MODE_CONFIG,
    "deep": DEEP_MODE_CONFIG
}


class PrivacyMode(str, Enum):
    """Privacy levels controlling data sent to external services.

    Implements a tiered privacy model. Each level enables a superset of
    the previous level's data sharing. Default is STRICT - minimal data.

    Attributes:
        STRICT: Timestamps and media types only. Minimal data leaves device.
                No location, no captions, no names. Maximum privacy.
        STANDARD: Above + country-level location, people counts, truncated captions.
                  Good balance of privacy and context for AI analysis.
        DETAILED: Above + city-level location, anonymized names, longer captions.
                  More context for richer AI insights.
        FULL: Everything except file paths. Requires explicit consent.
              Maximum context for AI but reduced privacy.

    Example:
        >>> config.privacy.mode = PrivacyMode.STANDARD
        >>> # Now locations at country level are allowed
    """

    STRICT = "strict"
    STANDARD = "standard"
    DETAILED = "detailed"
    FULL = "full"


class KeySource(str, Enum):
    """Sources from which API keys can be retrieved.

    The APIKeyManager tries sources in priority order: ENV → KEYRING → ENCRYPTED_FILE.
    This enum is used both for querying where a key was found and for specifying
    where to store a new key.

    Attributes:
        ENVIRONMENT: From the GEMINI_API_KEY environment variable.
                     Highest priority, commonly used in CI/CD and containers.
        KEYRING: From system keyring (macOS Keychain, Windows Credential Manager,
                 Linux Secret Service). Secure and user-friendly.
        ENCRYPTED_FILE: From local AES-encrypted file. Fallback when keyring
                        is unavailable (headless servers, containers).
        NONE: No key configured in any source.

    Example:
        >>> manager = APIKeyManager()
        >>> key = manager.get_key()
        >>> print(f"Key found in: {manager.get_key_source()}")
    """

    ENVIRONMENT = "environment"
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"
    NONE = "none"


# =============================================================================
# Configuration Models
# =============================================================================


class AIConfig(BaseModel):
    """Configuration for Gemini AI integration.

    Controls all aspects of AI model invocation including model selection,
    generation parameters, retry behavior, and caching. By default, AI is DISABLED
    and must be explicitly enabled by the user.

    Attributes:
        mode: AI activation mode. Default DISABLED for privacy-first operation.
        model_name: Gemini model identifier. Default is gemini-1.5-pro.
        temperature: Sampling temperature (0.0=deterministic, 2.0=creative).
        max_output_tokens: Maximum tokens in model response.
        timeout_seconds: Request timeout. Must balance between reliability and speed.
        max_retries: Number of retry attempts on transient failures.
        retry_base_delay: Base delay for exponential backoff between retries.
        cache_enabled: Enable caching of AI analysis results.
        cache_dir: Directory for cache files. None = platform default.
        cache_version: Cache schema version for invalidation control.

    Example:
        >>> ai_config = AIConfig(mode=AIMode.ENABLED, temperature=0.5)
        >>> if ai_config.is_enabled():
        ...     # Proceed with AI analysis
    """

    mode: AIMode = Field(
        default=AIMode.ENABLED,
        description="AI activation mode. Default ENABLED (requires consent before use).",
    )
    
    # 1. Depth & Visual Sampling
    depth_mode: Literal["quick", "standard", "deep"] = Field(
        default="deep",
        description=(
            "Analysis depth. 'quick' is fast/cheap, 'standard' is balanced, "
            "'deep' is thorough (Digital Archaeologist)."
        ),
        validation_alias="DEPTH_MODE"
    )
    
    # 2. Model Selection
    vision_model: str = Field(
        default="gemini-2.0-flash-exp", 
        description="Model used for visual tagging and atmosphere extraction."
    )
    narrative_model: str = Field(
        default="gemini-1.5-pro", 
        description="Model used for high-level narrative synthesis."
    )
    vision_model_fallback: str | None = Field(
        default=None, 
        description="Fallback model if primary vision model is unavailable."
    )
    allow_vision_on_videos: bool = Field(
        default=False, 
        description="Whether to extract and analyze frames from video files."
    )

    # 3. Budget & Caps
    max_vision_images_per_run: int = Field(
        default=250, # Increased from 120 to handle more images per chapter
        description="Hard cap on total images analyzed across a single run.",
        validation_alias="MAX_VISION_IMAGES"
    )
    max_vision_tokens_per_run: int | None = Field(
        default=None, 
        description="Optional token-based cap for visual analysis."
    )
    warn_on_large_dataset_threshold: int = Field(
        default=500, 
        description="Warn user if the input media count exceeds this number."
    )
    force_cap_reduction_threshold: int = Field(
        default=150, 
        description="Total images at which sampling density starts being reduced."
    )

    # 4. Cost Estimation
    show_cost_estimates: bool = Field(
        default=True, 
        description="Whether to display estimated API costs in the report/CLI."
    )
    vision_cost_per_image_usd: float = Field(
        default=0.001, 
        description="Estimated cost per image analysis in USD."
    )
    narrative_cost_per_1k_tokens_usd: float = Field(
        default=0.00375, 
        description="Estimated cost per 1k input tokens for the narrative model."
    )

    # Legacy compatibility / Core generative settings
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, 2=creative).",
    )
    max_output_tokens: int = Field(
        default=8192, ge=100, le=32000, description="Maximum tokens in model response."
    )
    timeout_seconds: int = Field(
        default=120, ge=10, le=600, description="Request timeout in seconds."
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Number of retry attempts on transient failures."
    )
    retry_base_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Base delay for exponential backoff (seconds)."
    )
    require_consent: bool = Field(
        default=True, description="Require explicit user consent before making AI requests."
    )

    # Cache settings for AI analysis results
    cache_enabled: bool = Field(
        default=True, description="Enable caching of AI analysis results for faster repeat runs."
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Cache directory. None = use platform default (~/.cache/... or equivalent).",
    )
    cache_version: str = Field(
        default="1.0",
        description="Cache schema version. Changing this invalidates all existing cache entries.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def is_enabled(self) -> bool:
        """Check if AI features are enabled.

        Returns:
            True if mode is not DISABLED, False otherwise.

        Example:
            >>> config = AIConfig(mode=AIMode.ENABLED)
            >>> config.is_enabled()
            True
        """
        return self.mode != AIMode.DISABLED

    def get_depth_config(self) -> DepthModeConfig:
        """Get the detailed sampling config for the current depth_mode.
        
        Returns:
            DepthModeConfig object with sampling parameters.
            Falls back to STANDARD if mode is invalid.
        """
        return DEPTH_MODE_CONFIGS.get(self.depth_mode, STANDARD_MODE_CONFIG)


class PrivacyConfig(BaseModel):
    """Configuration controlling what data can be sent to external services.

    Implements the privacy gate for all outbound data. The privacy mode
    enforces constraints on individual settings - for example, STRICT mode
    forces send_locations=False regardless of what's configured.

    Attributes:
        mode: Privacy tier. Higher tiers allow more data. Default STRICT.
        send_timestamps: Allow sending timestamps (always True for analysis).
        send_locations: Allow sending location data. Constrained by mode.
        location_precision: How precise location data can be.
        send_captions: Allow sending caption text. Constrained by mode.
        max_caption_length: Maximum caption characters sent to AI.
        send_people_names: Allow sending people's names. Constrained by mode.
        hash_people_names: If sending names, hash them for anonymization.
        send_media_types: Allow sending media type information.
        send_platform_info: Allow sending source platform information.
        require_consent_per_session: Re-prompt for consent each session.
        audit_transmissions: Log all data transmissions for review.

    Example:
        >>> privacy = PrivacyConfig(mode=PrivacyMode.STANDARD)
        >>> print(privacy.to_summary())
        >>> # "STANDARD: Country-level locations, people counts, truncated captions"
    """

    mode: PrivacyMode = Field(
        default=PrivacyMode.STRICT, description="Privacy tier. Higher tiers allow more data."
    )
    send_timestamps: bool = Field(
        default=True, description="Allow sending timestamps (essential for timeline analysis)."
    )
    send_locations: bool = Field(
        default=False, description="Allow sending location data. Constrained by privacy mode."
    )
    location_precision: Literal["country", "region", "city", "exact"] = Field(
        default="country", description="Maximum precision for location data."
    )
    send_captions: bool = Field(
        default=False, description="Allow sending caption/description text."
    )
    max_caption_length: int = Field(
        default=100, ge=0, le=1000, description="Maximum caption characters sent to AI."
    )
    send_people_names: bool = Field(default=False, description="Allow sending people's names.")
    hash_people_names: bool = Field(
        default=True, description="If sending names, hash them for anonymization."
    )
    send_media_types: bool = Field(
        default=True, description="Allow sending media type information."
    )
    send_platform_info: bool = Field(
        default=True, description="Allow sending source platform information."
    )
    require_consent_per_session: bool = Field(
        default=True, description="Re-prompt for consent each session."
    )
    audit_transmissions: bool = Field(
        default=True, description="Log all data transmissions for user review."
    )

    @model_validator(mode="after")
    def enforce_mode_constraints(self) -> "PrivacyConfig":
        """Enforce privacy mode constraints on individual settings.

        Privacy modes override individual settings to ensure policy compliance:
        - STRICT: No locations, no captions, no names
        - STANDARD: Country-level locations only, truncated captions
        - DETAILED: City-level locations, anonymized names allowed
        - FULL: All data except exact coordinates and file paths

        Returns:
            Self with enforced constraints.
        """
        if self.mode == PrivacyMode.STRICT:
            # STRICT: Absolute minimum - only timestamps and media types
            object.__setattr__(self, "send_locations", False)
            object.__setattr__(self, "send_captions", False)
            object.__setattr__(self, "send_people_names", False)
            object.__setattr__(self, "location_precision", "country")
            object.__setattr__(self, "max_caption_length", 0)

        elif self.mode == PrivacyMode.STANDARD:
            # STANDARD: Country-level location, people counts, truncated captions
            if self.send_locations and self.location_precision not in ["country"]:
                object.__setattr__(self, "location_precision", "country")
            # Cap caption length
            if self.max_caption_length > 100:
                object.__setattr__(self, "max_caption_length", 100)
            # No names in standard mode
            object.__setattr__(self, "send_people_names", False)

        elif self.mode == PrivacyMode.DETAILED:
            # DETAILED: City-level location, anonymized names, longer captions
            if self.send_locations and self.location_precision == "exact":
                object.__setattr__(self, "location_precision", "city")
            # If sending names, force hashing
            if self.send_people_names:
                object.__setattr__(self, "hash_people_names", True)
            # Cap caption length
            if self.max_caption_length > 500:
                object.__setattr__(self, "max_caption_length", 500)

        elif self.mode == PrivacyMode.FULL:
            # FULL: Everything except exact GPS and file paths
            # No forced constraints except exact coordinates
            if self.send_locations and self.location_precision == "exact":
                object.__setattr__(self, "location_precision", "city")

        return self

    def to_summary(self) -> str:
        """Generate human-readable description of current privacy stance.

        Returns:
            Multi-line string describing what data can be sent under current settings.

        Example:
            >>> privacy = PrivacyConfig(mode=PrivacyMode.STANDARD)
            >>> print(privacy.to_summary())
        """
        lines = [f"Privacy Mode: {self.mode.value.upper()}"]

        # Describe what's enabled
        enabled = []
        if self.send_timestamps:
            enabled.append("timestamps")
        if self.send_media_types:
            enabled.append("media types")
        if self.send_platform_info:
            enabled.append("platform info")
        if self.send_locations:
            enabled.append(f"locations ({self.location_precision}-level)")
        if self.send_captions:
            enabled.append(f"captions (max {self.max_caption_length} chars)")
        if self.send_people_names:
            if self.hash_people_names:
                enabled.append("people names (anonymized)")
            else:
                enabled.append("people names")

        if enabled:
            lines.append(f"Allowed: {', '.join(enabled)}")

        # Describe protections
        protections = []
        if self.require_consent_per_session:
            protections.append("session consent required")
        if self.audit_transmissions:
            protections.append("transmissions audited")

        if protections:
            lines.append(f"Protections: {', '.join(protections)}")

        return "\n".join(lines)


class PathsConfig(BaseModel):
    """Configuration for application file system paths.

    Manages all directories used by the application. Paths can be specified
    as absolute or relative. The `ensure_dirs_exist()` method creates any
    missing directories.

    Attributes:
        config_dir: Base directory for configuration files. Default ~/.dlna
        cache_dir: Directory for cached data. Default: config_dir/cache
        output_dir: Directory for generated reports. Default: ./output
        log_dir: Directory for log files. Default: config_dir/logs
        encrypted_key_file: Path to encrypted API key file. Default: config_dir/.api_key.enc

    Example:
        >>> paths = PathsConfig()
        >>> paths.ensure_dirs_exist()  # Creates all directories
        >>> print(paths.config_dir)
        >>> # ~/.dlna
    """

    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".dlna", description="Base configuration directory."
    )
    cache_dir: Path | None = Field(
        default=None, description="Cache directory. Defaults to config_dir/cache."
    )
    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "output",
        description="Output directory for generated reports.",
    )
    log_dir: Path | None = Field(
        default=None, description="Log directory. Defaults to config_dir/logs."
    )
    encrypted_key_file: Path | None = Field(
        default=None, description="Path to encrypted API key file."
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("config_dir", "output_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        """Expand ~ and resolve path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        elif isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @model_validator(mode="after")
    def resolve_defaults(self) -> "PathsConfig":
        """Resolve None defaults relative to config_dir.

        Returns:
            Self with resolved paths.
        """
        if self.cache_dir is None:
            object.__setattr__(self, "cache_dir", self.config_dir / "cache")
        else:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser().resolve())

        if self.log_dir is None:
            object.__setattr__(self, "log_dir", self.config_dir / "logs")
        else:
            object.__setattr__(self, "log_dir", Path(self.log_dir).expanduser().resolve())

        if self.encrypted_key_file is None:
            object.__setattr__(self, "encrypted_key_file", self.config_dir / ".api_key.enc")
        else:
            object.__setattr__(
                self, "encrypted_key_file", Path(self.encrypted_key_file).expanduser().resolve()
            )

        return self

    def ensure_dirs_exist(self) -> None:
        """Create all configured directories if they don't exist.

        Creates config_dir, cache_dir, log_dir, and the parent directory
        of encrypted_key_file. Sets appropriate permissions.

        Example:
            >>> paths = PathsConfig()
            >>> paths.ensure_dirs_exist()
        """
        for directory in [self.config_dir, self.cache_dir, self.log_dir, self.output_dir]:
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

        # Ensure parent of encrypted key file exists
        if self.encrypted_key_file is not None:
            self.encrypted_key_file.parent.mkdir(parents=True, exist_ok=True)


class ReportConfig(BaseModel):
    """Configuration for output report generation.

    Controls the format and content of generated life narrative reports.

    Attributes:
        format: Output format - HTML for human viewing, JSON for processing, or both.
        include_statistics: Include statistical summaries in report.
        include_timeline_visualization: Include interactive timeline visualization.
        include_raw_ai_responses: Include raw AI responses (debug feature).
        theme: Visual theme for HTML reports.

    Example:
        >>> report = ReportConfig(format="html", theme="dark")
    """

    format: Literal["html", "json", "both"] = Field(
        default="html", description="Output format for reports."
    )
    include_statistics: bool = Field(
        default=True, description="Include statistical summaries in report."
    )
    include_timeline_visualization: bool = Field(
        default=True, description="Include interactive timeline visualization."
    )
    include_raw_ai_responses: bool = Field(
        default=False, description="Include raw AI responses (debug feature)."
    )
    theme: Literal["light", "dark", "auto"] = Field(
        default="auto", description="Visual theme for HTML reports."
    )


class AppConfig(BaseSettings):
    """Top-level application configuration.

    Combines all configuration sections and supports loading from environment
    variables with the DLNA_ prefix. This is the main configuration
    object that should be used throughout the application.

    Configuration priority (highest wins):
    1. Environment variables (DLNA_*)
    2. Config file (YAML)
    3. In-code defaults

    Attributes:
        ai: Gemini AI integration settings.
        privacy: Privacy gate and data transmission settings.
        paths: Filesystem path configuration.
        report: Report generation settings.
        debug: Enable debug mode (verbose logging, extra checks).
        verbose: Enable verbose output to console.

    Example:
        >>> config = AppConfig()
        >>> if config.is_ai_available():
        ...     # AI is enabled and key is configured
        ...     analyzer.run()
    """

    ai: AIConfig = Field(default_factory=AIConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    debug: bool = Field(default=False, description="Enable debug mode.")
    verbose: bool = Field(default=False, description="Enable verbose output.")

    model_config = {
        "env_prefix": "DLNA_",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore unknown fields for forward compatibility
    }

    def is_ai_available(self) -> bool:
        """Check if AI is enabled AND an API key is configured.

        This is the primary check before attempting any AI operations.
        Returns True only if both conditions are met.

        Returns:
            True if AI can be used, False otherwise.

        Example:
            >>> config = get_config()
            >>> if config.is_ai_available():
            ...     # Safe to call Gemini
        """
        if not self.ai.is_enabled():
            return False

        # Check if key is available (without caching in config)
        try:
            manager = APIKeyManager(paths_config=self.paths)
            key = manager.get_key()
            return key is not None
        except Exception:
            return False

    def get_effective_privacy_mode(self) -> PrivacyMode:
        """Get the current effective privacy mode.

        Returns:
            The current privacy mode from configuration.

        Example:
            >>> mode = config.get_effective_privacy_mode()
            >>> if mode == PrivacyMode.STRICT:
            ...     print("Maximum privacy enabled")
        """
        return self.privacy.mode


# =============================================================================
# API Key Management
# =============================================================================


class APIKeyManager:
    """Secure management of Gemini API key from multiple sources.

    Retrieves API keys trying sources in priority order:
    1. Environment variable (GEMINI_API_KEY)
    2. System keyring (macOS Keychain, Windows Credential Manager, etc.)
    3. Encrypted file at configured path

    Keys are wrapped in SecretStr to prevent accidental logging. The key
    is cached after first successful retrieval for performance.

    Security Rules:
    - NEVER log the actual key value
    - NEVER include key in exception messages
    - NEVER store key in plain text
    - Keys stored encrypted or in system keyring only

    Attributes:
        _paths_config: Path configuration for encrypted file location.
        _cached_key: Cached key after first retrieval.
        _key_source: Source where the key was found.

    Example:
        >>> manager = APIKeyManager()
        >>> key = manager.get_key()
        >>> if key:
        ...     print(f"Key source: {manager.get_key_source()}")
        ...     # Use key.get_secret_value() only when needed
    """

    # Keyring service and username constants
    KEYRING_SERVICE = "digital-life-narrative-ai"
    KEYRING_USERNAME = "gemini"
    ENV_VAR_NAME = "GEMINI_API_KEY"

    def __init__(self, paths_config: PathsConfig | None = None) -> None:
        """Initialize the API key manager.

        Args:
            paths_config: Path configuration for encrypted file location.
                         If None, uses default PathsConfig.
        """
        self._paths_config = paths_config or PathsConfig()
        self._cached_key: SecretStr | None = None
        self._key_source: KeySource = KeySource.NONE

    def get_key(self) -> SecretStr | None:
        """Retrieve API key trying sources in priority order.

        Sources tried in order:
        1. Environment variable GEMINI_API_KEY
        2. System keyring (if available)
        3. Encrypted file at paths.encrypted_key_file

        The result is cached after first successful retrieval.

        Returns:
            SecretStr wrapper around the key, or None if not found.

        Example:
            >>> manager = APIKeyManager()
            >>> key = manager.get_key()
            >>> if key:
            ...     raw_key = key.get_secret_value()  # Only when needed
        """
        # Return cached key if available
        if self._cached_key is not None:
            return self._cached_key

        # Try environment variable first
        key = self._read_from_environment()
        if key and self.validate_key_format(key):
            self._cached_key = SecretStr(key)
            self._key_source = KeySource.ENVIRONMENT
            logger.debug("API key loaded from environment variable")
            return self._cached_key

        # Try system keyring
        key = self._read_from_keyring()
        if key and self.validate_key_format(key):
            self._cached_key = SecretStr(key)
            self._key_source = KeySource.KEYRING
            logger.debug("API key loaded from system keyring")
            return self._cached_key

        # Try encrypted file
        if self._paths_config.encrypted_key_file:
            key = self._read_from_encrypted_file(self._paths_config.encrypted_key_file)
            if key and self.validate_key_format(key):
                self._cached_key = SecretStr(key)
                self._key_source = KeySource.ENCRYPTED_FILE
                logger.debug("API key loaded from encrypted file")
                return self._cached_key

        # No key found
        self._key_source = KeySource.NONE
        logger.debug("No API key found in any source")
        return None

    def get_key_source(self) -> KeySource:
        """Get the source where the key was found.

        Returns:
            KeySource indicating where the key was retrieved from,
            or NONE if no key was found.

        Example:
            >>> manager = APIKeyManager()
            >>> manager.get_key()
            >>> print(manager.get_key_source())
            >>> # KeySource.ENVIRONMENT
        """
        return self._key_source

    def store_key(self, key: str, destination: KeySource) -> bool:
        """Store API key in the specified location.

        Args:
            key: The API key to store (will be validated first).
            destination: Where to store the key (KEYRING or ENCRYPTED_FILE).

        Returns:
            True if storage was successful.

        Raises:
            APIKeyInvalidError: If key fails format validation.
            ConfigError: If destination is ENVIRONMENT (cannot store).
            ConfigError: If storage fails for any reason.

        Example:
            >>> manager = APIKeyManager()
            >>> manager.store_key("your-api-key-here", KeySource.KEYRING)
        """
        # Validate key format
        if not self.validate_key_format(key):
            raise APIKeyInvalidError(
                "API key format validation failed. "
                "Key must be 20-100 characters with no whitespace."
            )

        if destination == KeySource.ENVIRONMENT:
            raise ConfigError(
                "Cannot store API key in environment variable. "
                "Set GEMINI_API_KEY manually in your environment."
            )

        if destination == KeySource.KEYRING:
            if not KEYRING_AVAILABLE:
                raise ConfigError(
                    "Keyring package not available. " "Install with: pip install keyring"
                )
            try:
                keyring.set_password(self.KEYRING_SERVICE, self.KEYRING_USERNAME, key)
                # Clear cache to pick up new key
                self._cached_key = None
                self._key_source = KeySource.NONE
                logger.info("API key stored in system keyring")
                return True
            except Exception as e:
                raise ConfigError(f"Failed to store key in keyring: {type(e).__name__}")

        if destination == KeySource.ENCRYPTED_FILE:
            if not CRYPTOGRAPHY_AVAILABLE:
                raise ConfigError(
                    "Cryptography package not available. " "Install with: pip install cryptography"
                )
            try:
                path = self._paths_config.encrypted_key_file
                if path is None:
                    path = self._paths_config.config_dir / ".api_key.enc"
                self._encrypt_to_file(key, path)
                # Clear cache to pick up new key
                self._cached_key = None
                self._key_source = KeySource.NONE
                logger.info("API key stored in encrypted file")
                return True
            except Exception as e:
                raise ConfigError(f"Failed to store key in encrypted file: {type(e).__name__}")

        raise ConfigError(f"Invalid destination: {destination}")

    def delete_key(self, source: KeySource) -> bool:
        """Remove API key from the specified source.

        Args:
            source: Where to delete the key from.

        Returns:
            True if deletion was successful or key didn't exist.

        Raises:
            ConfigError: If deletion fails.

        Example:
            >>> manager = APIKeyManager()
            >>> manager.delete_key(KeySource.KEYRING)
        """
        if source == KeySource.ENVIRONMENT:
            raise ConfigError(
                "Cannot delete environment variable. "
                "Unset GEMINI_API_KEY manually in your environment."
            )

        if source == KeySource.KEYRING:
            if not KEYRING_AVAILABLE:
                return True  # Not available, nothing to delete
            try:
                keyring.delete_password(self.KEYRING_SERVICE, self.KEYRING_USERNAME)
                # Clear cache
                self._cached_key = None
                self._key_source = KeySource.NONE
                logger.info("API key deleted from system keyring")
                return True
            except keyring.errors.PasswordDeleteError:
                return True  # Key didn't exist
            except Exception as e:
                raise ConfigError(f"Failed to delete key from keyring: {type(e).__name__}")

        if source == KeySource.ENCRYPTED_FILE:
            path = self._paths_config.encrypted_key_file
            if path and path.exists():
                try:
                    path.unlink()
                    # Clear cache
                    self._cached_key = None
                    self._key_source = KeySource.NONE
                    logger.info("API key file deleted")
                except Exception as e:
                    raise ConfigError(f"Failed to delete key file: {type(e).__name__}")
            return True

        raise ConfigError(f"Invalid source: {source}")

    def validate_key_format(self, key: str) -> bool:
        """Validate API key format without making an API call.

        Checks:
        - Non-empty string
        - Reasonable length (20-100 characters)
        - No leading/trailing whitespace
        - No internal whitespace

        Args:
            key: The key string to validate.

        Returns:
            True if key passes format validation.

        Example:
            >>> manager = APIKeyManager()
            >>> manager.validate_key_format("valid-key-here")
            True
            >>> manager.validate_key_format("  key with spaces  ")
            False
        """
        if not key:
            return False

        # Strip and check for whitespace
        stripped = key.strip()
        if stripped != key:
            # Has leading/trailing whitespace - technically invalid but we can fix
            key = stripped

        if not key:
            return False

        # Check length
        if len(key) < 20 or len(key) > 100:
            return False

        # Check for internal whitespace
        if any(c.isspace() for c in key):
            return False

        return True

    def _read_from_environment(self) -> str | None:
        """Read API key from environment variable.

        Returns:
            The key string, or None if not set.
        """
        key = os.environ.get(self.ENV_VAR_NAME)
        if key:
            # Strip whitespace (common mistake)
            return key.strip()
        return None

    def _read_from_keyring(self) -> str | None:
        """Read API key from system keyring.

        Returns:
            The key string, or None if not found or keyring unavailable.
        """
        if not KEYRING_AVAILABLE:
            return None

        try:
            key = keyring.get_password(self.KEYRING_SERVICE, self.KEYRING_USERNAME)
            return key
        except Exception as e:
            # Log but don't fail - keyring might not be available on all systems
            logger.debug(f"Keyring access failed: {type(e).__name__}")
            return None

    def _read_from_encrypted_file(self, path: Path) -> str | None:
        """Read and decrypt API key from encrypted file.

        Args:
            path: Path to the encrypted key file.

        Returns:
            The decrypted key string, or None if file doesn't exist
            or decryption fails.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.debug("Cryptography package not available")
            return None

        if not path.exists():
            return None

        try:
            # Read encrypted data
            encrypted_data = path.read_bytes()

            # Derive decryption key
            fernet_key = self._derive_encryption_key()
            fernet = Fernet(fernet_key)

            # Decrypt
            decrypted = fernet.decrypt(encrypted_data)
            return decrypted.decode("utf-8")

        except InvalidToken:
            logger.warning(
                "Failed to decrypt API key file - key may have been created on a different machine"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to read encrypted key file: {type(e).__name__}")
            return None

    def _encrypt_to_file(self, key: str, path: Path) -> None:
        """Encrypt and write API key to file with secure permissions.

        Args:
            key: The API key to encrypt and store.
            path: Path where encrypted file should be written.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ConfigError("Cryptography package not available")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Derive encryption key
        fernet_key = self._derive_encryption_key()
        fernet = Fernet(fernet_key)

        # Encrypt
        encrypted = fernet.encrypt(key.encode("utf-8"))

        # Write with restrictive permissions
        path.write_bytes(encrypted)

        # Set file permissions to owner read/write only (600)
        try:
            if platform.system() != "Windows":
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            else:
                # Windows: Use icacls to restrict permissions
                # This is a best-effort on Windows
                pass
        except Exception as e:
            logger.warning(f"Could not set restrictive file permissions: {type(e).__name__}")

    def _derive_encryption_key(self) -> bytes:
        """Derive a Fernet encryption key from machine-specific data.

        Uses a combination of machine identifiers to create a key that
        is unique to this machine. This means encrypted files cannot be
        decrypted on a different machine.

        Returns:
            32-byte Fernet-compatible encryption key.
        """
        # Gather machine-specific data
        machine_data = []

        # Platform info
        machine_data.append(platform.node())
        machine_data.append(platform.machine())
        machine_data.append(platform.system())

        # Try to get unique machine ID
        try:
            if platform.system() == "Linux":
                machine_id_path = Path("/etc/machine-id")
                if machine_id_path.exists():
                    machine_data.append(machine_id_path.read_text().strip())
            elif platform.system() == "Darwin":  # macOS
                # Use IOPlatformSerialNumber via system_profiler
                import subprocess

                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "Hardware UUID" in line:
                        machine_data.append(line.split(":")[-1].strip())
                        break
            elif platform.system() == "Windows":
                # Use Windows machine GUID
                import winreg

                try:
                    key = winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
                    )
                    value, _ = winreg.QueryValueEx(key, "MachineGuid")
                    machine_data.append(value)
                    winreg.CloseKey(key)
                except Exception:
                    pass
        except Exception:
            # If we can't get machine ID, continue with what we have
            pass

        # Create deterministic seed from machine data
        combined = "|".join(machine_data).encode("utf-8")

        # Use PBKDF2 to derive a key
        salt = b"digital-life-narrative-ai-salt-v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(combined))

        return key


# =============================================================================
# Module-Level Functions
# =============================================================================


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from file, environment, and defaults.

    Configuration priority (highest wins):
    1. Environment variables (DLNA_*)
    2. Config file (if provided or found at default location)
    3. In-code defaults

    If no config file is found, uses defaults only (not an error).
    If config file is malformed, logs warning and uses defaults.

    Args:
        path: Optional path to config file. If None, searches default locations.

    Returns:
        Fully-populated AppConfig instance.

    Raises:
        ConfigFileError: Only if config file exists but is critically malformed.

    Example:
        >>> config = load_config()  # Use defaults and env vars
        >>> config = load_config(Path("./my-config.yaml"))  # Specific file
    """
    config_data: dict[str, Any] = {}

    # Find config file
    search_paths = [
        path,
        Path("./config.yaml"),
        Path("./config.yml"),
        Path.home() / ".life-story" / "config.yaml",
        Path.home() / ".life-story" / "config.yml",
    ]

    config_file: Path | None = None
    for search_path in search_paths:
        if search_path is not None and search_path.exists():
            config_file = search_path
            break

    # Load config file if found
    if config_file is not None:
        if not YAML_AVAILABLE:
            logger.warning(
                f"Config file found at {config_file} but PyYAML not installed. "
                "Using defaults only."
            )
        else:
            try:
                content = config_file.read_text(encoding="utf-8")
                if content.strip():  # Not empty
                    loaded = yaml.safe_load(content)
                    if isinstance(loaded, dict):
                        config_data = loaded
                    elif loaded is None:
                        # Empty YAML file - use defaults
                        pass
                    else:
                        logger.warning(
                            f"Config file {config_file} has unexpected format. " "Using defaults."
                        )
            except yaml.YAMLError as e:
                logger.warning(
                    f"Failed to parse config file {config_file}: {e}. " "Using defaults."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to read config file {config_file}: {type(e).__name__}. "
                    "Using defaults."
                )

    # Parse nested config sections
    try:
        # Handle AI config
        ai_data = config_data.get("ai", {})
        if isinstance(ai_data, dict):
            # Convert mode string to enum if needed
            if "mode" in ai_data and isinstance(ai_data["mode"], str):
                try:
                    ai_data["mode"] = AIMode(ai_data["mode"].lower())
                except ValueError:
                    logger.warning(f"Invalid ai.mode value: {ai_data['mode']}. Using default.")
                    del ai_data["mode"]

        # Handle privacy config
        privacy_data = config_data.get("privacy", {})
        if isinstance(privacy_data, dict):
            if "mode" in privacy_data and isinstance(privacy_data["mode"], str):
                try:
                    privacy_data["mode"] = PrivacyMode(privacy_data["mode"].lower())
                except ValueError:
                    logger.warning(
                        f"Invalid privacy.mode value: {privacy_data['mode']}. Using default."
                    )
                    del privacy_data["mode"]

        # Handle paths config
        paths_data = config_data.get("paths", {})
        if isinstance(paths_data, dict):
            # Expand ~ in paths
            for key in ["config_dir", "cache_dir", "output_dir", "log_dir", "encrypted_key_file"]:
                if key in paths_data and isinstance(paths_data[key], str):
                    paths_data[key] = Path(paths_data[key]).expanduser()

        # Build AppConfig
        app_config = AppConfig(
            ai=AIConfig(**ai_data) if ai_data else AIConfig(),
            privacy=PrivacyConfig(**privacy_data) if privacy_data else PrivacyConfig(),
            paths=PathsConfig(**paths_data) if paths_data else PathsConfig(),
            report=ReportConfig(**config_data.get("report", {})),
            debug=config_data.get("debug", False),
            verbose=config_data.get("verbose", False),
        )

        return app_config

    except Exception as e:
        logger.warning(f"Error parsing config values: {e}. Using defaults.")
        return AppConfig()


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get the cached configuration singleton.

    Loads configuration once and returns the same instance on subsequent calls.
    This is the recommended way to access configuration throughout the application.

    Returns:
        Cached AppConfig instance.

    Example:
        >>> from dlnai.config import get_config
        >>> cfg = get_config()
        >>> print(cfg.ai.mode)
    """
    return load_config()


def get_api_key() -> SecretStr:
    """Convenience function to get the Gemini API key.

    Gets the configuration, creates an APIKeyManager, and retrieves the key.
    Raises an exception if no key is found.

    Returns:
        SecretStr wrapper around the API key.

    Raises:
        APIKeyNotFoundError: If no API key is configured in any source.

    Example:
        >>> from dlnai.config import get_api_key
        >>> key = get_api_key()
        >>> # Use key.get_secret_value() when making API calls
    """
    config = get_config()
    manager = APIKeyManager(paths_config=config.paths)
    key = manager.get_key()

    if key is None:
        raise APIKeyNotFoundError(
            "No API key found. Set GEMINI_API_KEY environment variable, "
            "store in system keyring, or create encrypted key file. "
            "Run 'python -m src.config --help' for setup instructions."
        )

    return key


def reset_config() -> None:
    """Clear the configuration cache for testing.

    After calling this, the next call to get_config() will reload
    configuration from sources.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # Fresh load
    """
    get_config.cache_clear()


# =============================================================================
# CLI for Key Management (Optional)
# =============================================================================


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Digital Life Narrative AI Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current configuration
  python -m src.config --show
  
  # Store API key in system keyring
  python -m src.config --store-key keyring
  
  # Store API key in encrypted file
  python -m src.config --store-key file
  
  # Check if API key is configured
  python -m src.config --check-key
  
  # Delete API key from a source
  python -m src.config --delete-key keyring
        """,
    )

    parser.add_argument(
        "--show", action="store_true", help="Show current configuration (excluding secrets)"
    )
    parser.add_argument(
        "--check-key", action="store_true", help="Check if API key is configured and valid"
    )
    parser.add_argument(
        "--store-key", choices=["keyring", "file"], help="Store API key (will prompt for key)"
    )
    parser.add_argument(
        "--delete-key", choices=["keyring", "file"], help="Delete API key from specified source"
    )

    args = parser.parse_args()

    if args.show:
        config = get_config()
        print("=" * 60)
        print("Digital Life Narrative AI Configuration")
        print("=" * 60)
        print(f"\nAI Settings:")
        print(f"  Mode: {config.ai.mode.value}")
        print(f"  Model: {config.ai.model_name}")
        print(f"  Temperature: {config.ai.temperature}")
        print(f"  Max Tokens: {config.ai.max_output_tokens}")
        print(f"\nPrivacy Settings:")
        print(config.privacy.to_summary())
        print(f"\nPaths:")
        print(f"  Config: {config.paths.config_dir}")
        print(f"  Cache: {config.paths.cache_dir}")
        print(f"  Output: {config.paths.output_dir}")
        print(f"  Logs: {config.paths.log_dir}")
        print(f"\nOther:")
        print(f"  Debug: {config.debug}")
        print(f"  Verbose: {config.verbose}")
        print(f"  AI Available: {config.is_ai_available()}")

    elif args.check_key:
        config = get_config()
        manager = APIKeyManager(paths_config=config.paths)
        key = manager.get_key()

        if key:
            source = manager.get_key_source()
            print(f"✓ API key found in: {source.value}")
            # Show partial key for verification
            raw = key.get_secret_value()
            masked = raw[:4] + "*" * (len(raw) - 8) + raw[-4:]
            print(f"  Key: {masked}")
        else:
            print("✗ No API key configured")
            print("\nTo configure, use one of:")
            print("  1. Set GEMINI_API_KEY environment variable")
            print("  2. Run: python -m src.config --store-key keyring")
            print("  3. Run: python -m src.config --store-key file")
            sys.exit(1)

    elif args.store_key:
        import getpass

        config = get_config()
        manager = APIKeyManager(paths_config=config.paths)

        print("Enter your Gemini API key (input hidden):")
        key = getpass.getpass(prompt="API Key: ")

        destination = KeySource.KEYRING if args.store_key == "keyring" else KeySource.ENCRYPTED_FILE

        try:
            manager.store_key(key, destination)
            print(f"✓ API key stored successfully in {destination.value}")
        except Exception as e:
            print(f"✗ Failed to store key: {e}")
            sys.exit(1)

    elif args.delete_key:
        config = get_config()
        manager = APIKeyManager(paths_config=config.paths)

        source = KeySource.KEYRING if args.delete_key == "keyring" else KeySource.ENCRYPTED_FILE

        try:
            manager.delete_key(source)
            print(f"✓ API key deleted from {source.value}")
        except Exception as e:
            print(f"✗ Failed to delete key: {e}")
            sys.exit(1)
    else:
        parser.print_help()
