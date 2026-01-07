"""Configuration and API key management for Digital Life Narrative AI.

This module handles all configuration, API key management, and privacy settings.
Security is critical because users are trusting us with their personal media metadata.

Security notes:
- API keys are never logged or printed
- Encrypted file backend uses Fernet symmetric encryption
- Keyring backend leverages OS-level credential storage
"""

import base64
import hashlib
import os
import platform
import secrets
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, field_validator

from organizer.models import SourcePlatform


# =============================================================================
# Exceptions
# =============================================================================


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""

    pass


class APIKeyNotFoundError(Exception):
    """Raised when API key is not configured or cannot be retrieved."""

    pass


# =============================================================================
# Enums
# =============================================================================


class KeyStorageBackend(str, Enum):
    """Backend options for storing API keys securely.

    Attributes:
        ENV: Store in environment variable (GEMINI_API_KEY)
        KEYRING: Use system keyring (OS credential manager)
        ENCRYPTED_FILE: Store in Fernet-encrypted local file
    """

    ENV = "env"
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"


# =============================================================================
# Configuration Models
# =============================================================================


class PrivacySettings(BaseModel):
    """Privacy settings for controlling what data is sent to AI.

    These settings help protect user privacy by limiting or anonymizing
    the data that gets sent to external AI services.

    Attributes:
        anonymize_paths: Strip full file paths before sending to AI
        truncate_captions: Maximum characters to send for captions
        exclude_platforms: Platforms to skip entirely in analysis
        hash_people_names: Replace people names with hashes
        local_only_mode: If True, refuse all AI API calls
    """

    anonymize_paths: bool = True
    truncate_captions: int = Field(default=200, ge=10, le=2000)
    exclude_platforms: list[SourcePlatform] = Field(default_factory=list)
    hash_people_names: bool = False
    local_only_mode: bool = False


class AISettings(BaseModel):
    """Settings for AI model configuration.

    Controls the behavior of the Gemini API client and response generation.

    Attributes:
        model_name: The Gemini model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-2.0)
        timeout_seconds: Request timeout
        max_retries: Number of retry attempts on failure
    """

    model_name: str = "gemini-1.5-pro"
    max_tokens: int = Field(default=8000, ge=100, le=100000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)


class AppConfig(BaseModel):
    """Main application configuration.

    Central configuration class that holds all settings for the application.
    Can be loaded from and saved to YAML files.

    Attributes:
        ai: AI/Gemini settings
        privacy: Privacy and data handling settings
        key_storage_backend: How API keys are stored
        encrypted_key_file_path: Path to encrypted key file (if using that backend)
        output_format: Format for generated reports
    """

    ai: AISettings = Field(default_factory=AISettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    key_storage_backend: KeyStorageBackend = KeyStorageBackend.ENV
    encrypted_key_file_path: Path | None = None
    output_format: Literal["html", "json", "both"] = "html"

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the platform-appropriate default configuration path.

        Returns:
            Path to the default config file location:
            - Windows: %APPDATA%/digital-life-narrative/config.yaml
            - macOS: ~/Library/Application Support/digital-life-narrative/config.yaml
            - Linux: ~/.config/digital-life-narrative/config.yaml
        """
        system = platform.system()

        if system == "Windows":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        elif system == "Darwin":  # macOS
            base = Path.home() / "Library" / "Application Support"
        else:  # Linux and others
            xdg_config = os.environ.get("XDG_CONFIG_HOME")
            base = Path(xdg_config) if xdg_config else Path.home() / ".config"

        return base / "digital-life-narrative" / "config.yaml"

    @classmethod
    def load_from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Loaded AppConfig instance.

        Raises:
            ConfigurationError: If file cannot be read or parsed.
        """
        try:
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                return cls()

            # Handle nested models
            if "ai" in data and isinstance(data["ai"], dict):
                data["ai"] = AISettings(**data["ai"])
            if "privacy" in data and isinstance(data["privacy"], dict):
                # Convert platform strings to enums
                if "exclude_platforms" in data["privacy"]:
                    data["privacy"]["exclude_platforms"] = [
                        SourcePlatform(p) if isinstance(p, str) else p
                        for p in data["privacy"]["exclude_platforms"]
                    ]
                data["privacy"] = PrivacySettings(**data["privacy"])
            if "key_storage_backend" in data:
                data["key_storage_backend"] = KeyStorageBackend(data["key_storage_backend"])
            if "encrypted_key_file_path" in data and data["encrypted_key_file_path"]:
                data["encrypted_key_file_path"] = Path(data["encrypted_key_file_path"])

            return cls(**data)

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def save_to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file.

        Creates parent directories if they don't exist.

        Args:
            path: Path to save the configuration file.

        Raises:
            ConfigurationError: If file cannot be written.
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable dict
            data = {
                "ai": {
                    "model_name": self.ai.model_name,
                    "max_tokens": self.ai.max_tokens,
                    "temperature": self.ai.temperature,
                    "timeout_seconds": self.ai.timeout_seconds,
                    "max_retries": self.ai.max_retries,
                },
                "privacy": {
                    "anonymize_paths": self.privacy.anonymize_paths,
                    "truncate_captions": self.privacy.truncate_captions,
                    "exclude_platforms": [p.value for p in self.privacy.exclude_platforms],
                    "hash_people_names": self.privacy.hash_people_names,
                    "local_only_mode": self.privacy.local_only_mode,
                },
                "key_storage_backend": self.key_storage_backend.value,
                "encrypted_key_file_path": (
                    str(self.encrypted_key_file_path) if self.encrypted_key_file_path else None
                ),
                "output_format": self.output_format,
            }

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")


# =============================================================================
# API Key Manager
# =============================================================================


class APIKeyManager:
    """Secure manager for API key storage and retrieval.

    Supports multiple storage backends:
    - ENV: Environment variable (GEMINI_API_KEY)
    - KEYRING: OS-level credential storage
    - ENCRYPTED_FILE: Fernet-encrypted local file

    Security:
    - Keys are never logged or printed
    - Encrypted storage uses machine-derived encryption keys
    - Key validation without exposing content

    Attributes:
        backend: The storage backend to use
        encrypted_file_path: Path to encrypted key file (for ENCRYPTED_FILE backend)
    """

    SERVICE_NAME = "digital-life-narrative-ai"
    ENV_VAR_NAME = "GEMINI_API_KEY"
    MIN_KEY_LENGTH = 10
    MAX_KEY_LENGTH = 256

    def __init__(
        self,
        backend: KeyStorageBackend,
        encrypted_file_path: Path | None = None,
    ) -> None:
        """Initialize the API key manager.

        Args:
            backend: Storage backend to use.
            encrypted_file_path: Path for encrypted file storage.
                Required if backend is ENCRYPTED_FILE.

        Raises:
            ConfigurationError: If encrypted file backend selected without path.
        """
        self.backend = backend
        self.encrypted_file_path = encrypted_file_path

        if backend == KeyStorageBackend.ENCRYPTED_FILE and not encrypted_file_path:
            raise ConfigurationError("encrypted_file_path required for ENCRYPTED_FILE backend")

    def _validate_key_format(self, key: str) -> None:
        """Validate API key format without exposing the key.

        Args:
            key: The API key to validate.

        Raises:
            ConfigurationError: If key format is invalid.
        """
        if not key or not isinstance(key, str):
            raise ConfigurationError("API key must be a non-empty string")
        if len(key) < self.MIN_KEY_LENGTH:
            raise ConfigurationError(
                f"API key too short (minimum {self.MIN_KEY_LENGTH} characters)"
            )
        if len(key) > self.MAX_KEY_LENGTH:
            raise ConfigurationError(f"API key too long (maximum {self.MAX_KEY_LENGTH} characters)")
        if not key.strip() == key:
            raise ConfigurationError("API key should not have leading/trailing whitespace")

    def _get_machine_key(self) -> bytes:
        """Derive an encryption key from machine-specific data.

        Uses a combination of machine identifiers to create a reproducible
        encryption key unique to this machine.

        Returns:
            32-byte encryption key suitable for Fernet.
        """
        # Gather machine-specific identifiers
        identifiers = [
            platform.node(),  # Hostname
            platform.machine(),  # CPU architecture
            os.environ.get("USERNAME", os.environ.get("USER", "default")),
        ]

        # Create deterministic hash
        combined = ":".join(identifiers).encode("utf-8")
        key_bytes = hashlib.sha256(combined).digest()

        # Fernet requires URL-safe base64-encoded 32-byte key
        return base64.urlsafe_b64encode(key_bytes)

    def _get_fernet(self) -> Fernet:
        """Get Fernet instance for encryption/decryption.

        Returns:
            Configured Fernet instance.
        """
        return Fernet(self._get_machine_key())

    def store_key(self, key: str) -> None:
        """Store the Gemini API key securely.

        Args:
            key: The API key to store.

        Raises:
            ConfigurationError: If key format is invalid or storage fails.
        """
        self._validate_key_format(key)

        if self.backend == KeyStorageBackend.ENV:
            # Note: This only sets for current process; user should set in shell
            os.environ[self.ENV_VAR_NAME] = key

        elif self.backend == KeyStorageBackend.KEYRING:
            try:
                import keyring

                keyring.set_password(self.SERVICE_NAME, "api_key", key)
            except Exception as e:
                raise ConfigurationError(f"Failed to store key in keyring: {e}")

        elif self.backend == KeyStorageBackend.ENCRYPTED_FILE:
            if not self.encrypted_file_path:
                raise ConfigurationError("Encrypted file path not configured")

            try:
                fernet = self._get_fernet()
                encrypted = fernet.encrypt(key.encode("utf-8"))

                # Ensure directory exists
                self.encrypted_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write with restrictive permissions (owner read/write only)
                self.encrypted_file_path.write_bytes(encrypted)

                # Set restrictive permissions on Unix systems
                if platform.system() != "Windows":
                    os.chmod(self.encrypted_file_path, 0o600)

            except Exception as e:
                raise ConfigurationError(f"Failed to store encrypted key: {e}")

    def retrieve_key(self) -> str | None:
        """Retrieve the stored API key.

        Returns:
            The API key if found, None otherwise.

        Raises:
            ConfigurationError: If decryption or retrieval fails.
        """
        if self.backend == KeyStorageBackend.ENV:
            return os.environ.get(self.ENV_VAR_NAME)

        elif self.backend == KeyStorageBackend.KEYRING:
            try:
                import keyring

                return keyring.get_password(self.SERVICE_NAME, "api_key")
            except Exception:
                return None

        elif self.backend == KeyStorageBackend.ENCRYPTED_FILE:
            if not self.encrypted_file_path or not self.encrypted_file_path.exists():
                return None

            try:
                fernet = self._get_fernet()
                encrypted = self.encrypted_file_path.read_bytes()
                decrypted = fernet.decrypt(encrypted)
                return decrypted.decode("utf-8")
            except InvalidToken:
                raise ConfigurationError("Failed to decrypt API key - encryption key mismatch")
            except Exception as e:
                raise ConfigurationError(f"Failed to retrieve encrypted key: {e}")

        return None

    def delete_key(self) -> None:
        """Remove the stored API key.

        Raises:
            ConfigurationError: If deletion fails.
        """
        if self.backend == KeyStorageBackend.ENV:
            if self.ENV_VAR_NAME in os.environ:
                del os.environ[self.ENV_VAR_NAME]

        elif self.backend == KeyStorageBackend.KEYRING:
            try:
                import keyring

                keyring.delete_password(self.SERVICE_NAME, "api_key")
            except Exception:
                pass  # Key may not exist

        elif self.backend == KeyStorageBackend.ENCRYPTED_FILE:
            if self.encrypted_file_path and self.encrypted_file_path.exists():
                try:
                    # Overwrite with random data before deletion for security
                    self.encrypted_file_path.write_bytes(secrets.token_bytes(64))
                    self.encrypted_file_path.unlink()
                except Exception as e:
                    raise ConfigurationError(f"Failed to delete encrypted key file: {e}")

    def is_key_configured(self) -> bool:
        """Check if an API key is configured.

        Returns:
            True if a key is stored and retrievable.
        """
        try:
            key = self.retrieve_key()
            return key is not None and len(key) >= self.MIN_KEY_LENGTH
        except ConfigurationError:
            return False


# =============================================================================
# Module-Level Functions
# =============================================================================


def get_config() -> AppConfig:
    """Load configuration from default path or return defaults.

    Attempts to load from the platform-specific default config path.
    If the file doesn't exist, returns default configuration.

    Returns:
        Loaded or default AppConfig instance.
    """
    config_path = AppConfig.get_default_config_path()

    if config_path.exists():
        try:
            return AppConfig.load_from_yaml(config_path)
        except ConfigurationError:
            # Fall back to defaults if config is corrupted
            return AppConfig()

    return AppConfig()


def configure_api_key(key: str, backend: KeyStorageBackend) -> None:
    """High-level function to store an API key.

    Convenience function for CLI usage to configure the API key.

    Args:
        key: The Gemini API key to store.
        backend: Storage backend to use.

    Raises:
        ConfigurationError: If storage fails.
        APIKeyNotFoundError: If key validation fails.
    """
    config = get_config()

    # Determine encrypted file path if needed
    encrypted_path = None
    if backend == KeyStorageBackend.ENCRYPTED_FILE:
        encrypted_path = config.encrypted_key_file_path
        if not encrypted_path:
            # Use default location next to config
            config_dir = AppConfig.get_default_config_path().parent
            encrypted_path = config_dir / "credentials.enc"

    manager = APIKeyManager(backend, encrypted_path)
    manager.store_key(key)

    # Update config with new backend setting
    config.key_storage_backend = backend
    if encrypted_path:
        config.encrypted_key_file_path = encrypted_path

    # Save updated config
    config.save_to_yaml(AppConfig.get_default_config_path())


def get_api_key() -> str:
    """Retrieve the configured API key.

    Convenience function to get the API key using current configuration.

    Returns:
        The API key.

    Raises:
        APIKeyNotFoundError: If no key is configured.
    """
    config = get_config()
    manager = APIKeyManager(
        config.key_storage_backend,
        config.encrypted_key_file_path,
    )

    key = manager.retrieve_key()
    if not key:
        raise APIKeyNotFoundError("No API key configured. Run 'organizer configure' to set up.")

    return key
