"""
Disclosure Handler for Pixel Analysis.

This module handles the disclosure acknowledgement flow for pixel analysis,
ensuring users explicitly acknowledge what data is sent to Gemini Vision
before the feature is activated. Users can also opt-out if they prefer.
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

from dlnai.core.safety import SafetySettings

logger = logging.getLogger(__name__)

# --- Disclosure Text ---

PIXEL_ANALYSIS_DISCLOSURE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PIXEL ANALYSIS DISCLOSURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI-powered content classification is enabled by default to help organize and
filter your personal media library.

What's Sent to Gemini Vision:
  • Small thumbnails (256x256px, reduced quality)
  • No metadata, filenames, or location data

How It's Used:
  • Classifies content for safety filtering
  • Helps you control what appears in reports
  • Data sent only to Google Gemini (no third parties)

Your Choice:
  [1] Continue with pixel analysis (recommended for best results)
  [2] Opt-out (use metadata-only detection instead)

You can change this setting anytime in your configuration.
More info: PRIVACY.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def show_disclosure_cli() -> Tuple[bool, bool]:
    """
    Show disclosure in CLI and get user acknowledgement.

    Returns:
        Tuple of (acknowledged, opted_out)
        - (True, False): User acknowledged, pixel analysis enabled
        - (False, True): User opted out, pixel analysis disabled
    """
    print(PIXEL_ANALYSIS_DISCLOSURE)

    while True:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            print("\n✓ Pixel analysis enabled.\n")
            return (True, False)
        elif choice == "2":
            print("\n✓ Using metadata-only detection.\n")
            return (False, True)
        else:
            print("Invalid input. Please enter 1 or 2.")


def acknowledge_disclosure(
    settings: SafetySettings,
    interactive: bool = True,
) -> SafetySettings:
    """
    Handle disclosure acknowledgement flow.

    Args:
        settings: Current safety settings
        interactive: If True, show interactive prompt. If False, just check status.

    Returns:
        Updated SafetySettings with acknowledgement/opt-out recorded
    """
    # Check if already acknowledged
    if settings.pixel_analysis_disclosure_acknowledged:
        logger.info("Pixel analysis disclosure already acknowledged")
        return settings

    # Check if user has opted out
    if not settings.use_pixel_analysis:
        logger.info("User has opted out of pixel analysis")
        return settings

    # Need acknowledgement
    if not interactive:
        logger.warning(
            "Pixel analysis requires disclosure acknowledgement but running in non-interactive mode"
        )
        return settings

    # Show disclosure and get acknowledgement
    acknowledged, opted_out = show_disclosure_cli()

    if acknowledged:
        settings.pixel_analysis_disclosure_acknowledged = True
        settings.pixel_analysis_disclosure_timestamp = datetime.now()
        settings.use_pixel_analysis = True
        logger.info("Pixel analysis disclosure acknowledged")
    elif opted_out:
        settings.use_pixel_analysis = False
        settings.pixel_analysis_disclosure_acknowledged = False
        logger.info("User opted out of pixel analysis")

    return settings


def check_and_prompt_disclosure(
    settings: Optional[SafetySettings] = None,
    interactive: bool = True,
) -> SafetySettings:
    """
    Check disclosure status and prompt if needed.

    This is the main entry point for the disclosure flow. Call this
    before running any pixel analysis operations.

    Args:
        settings: Current safety settings (creates default if None)
        interactive: If True, show interactive prompt when needed

    Returns:
        Updated SafetySettings with disclosure handled
    """
    if settings is None:
        settings = SafetySettings()

    # If pixel analysis is disabled (user opted out), no need to prompt
    if not settings.use_pixel_analysis:
        return settings

    # If already acknowledged, no need to prompt
    if settings.pixel_analysis_disclosure_acknowledged:
        return settings

    # Need to show disclosure
    logger.info("Pixel analysis disclosure required")
    return acknowledge_disclosure(settings, interactive=interactive)


def create_disclosure_html() -> str:
    """
    Create HTML version of disclosure for web UI.

    Returns:
        HTML string with disclosure and acknowledgement buttons
    """
    return """
    <div class="disclosure-modal">
        <div class="disclosure-content">
            <h2>Pixel Analysis Disclosure</h2>
            <p>AI-powered content classification helps organize and filter your personal media library.</p>
            
            <h3>What's Sent to Gemini Vision</h3>
            <ul>
                <li>Small thumbnails (256x256px, reduced quality)</li>
                <li>No metadata, filenames, or location data</li>
            </ul>
            
            <h3>How It's Used</h3>
            <ul>
                <li>Classifies content for safety filtering</li>
                <li>Helps you control what appears in reports</li>
                <li>Data sent only to Google Gemini (no third parties)</li>
            </ul>
            
            <div class="disclosure-actions">
                <button class="btn-primary" onclick="acknowledgeDisclosure()">
                    Continue with Pixel Analysis
                </button>
                <button class="btn-secondary" onclick="optOutDisclosure()">
                    Use Metadata-Only
                </button>
            </div>
            
            <p class="disclosure-footer">
                <small>You can change this anytime in settings. See <a href="PRIVACY.md">PRIVACY.md</a> for details.</small>
            </p>
        </div>
    </div>
    """


def get_disclosure_status_message(settings: SafetySettings) -> str:
    """
    Get a human-readable status message about disclosure.

    Args:
        settings: Safety settings to check

    Returns:
        Status message string
    """
    if not settings.use_pixel_analysis:
        return "Pixel analysis: OPTED OUT (metadata-only detection active)"

    if settings.pixel_analysis_disclosure_acknowledged:
        timestamp = settings.pixel_analysis_disclosure_timestamp
        if timestamp:
            return f"Pixel analysis: ENABLED (acknowledged on {timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
        return "Pixel analysis: ENABLED (acknowledged)"

    return "Pixel analysis: PENDING DISCLOSURE (awaiting user acknowledgement)"
