"""Command-line interface for Digital Life Narrative AI.

The CLI guides users through the AI-first experience, making it clear
that AI analysis is the primary value of the application.

Built with Click for commands and Rich for beautiful terminal output.

Usage:
    organizer analyze -i ~/exports -o ./my_story
    organizer config set-key
    organizer scan ~/exports
    organizer organize -i ~/exports -o ./organized --report ./my_story.json
"""

from __future__ import annotations

import asyncio
import logging
import sys
import webbrowser
from enum import Enum
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from organizer import __version__
from organizer.config import (
    APIKeyManager,
    AppConfig,
    PrivacySettings,
    get_config,
)
from organizer.detection import DetectionResult, detect_export_source
from organizer.models import AnalysisConfig, MediaItem, SourcePlatform
from organizer.parsers import parse_all_sources
from organizer.report import ReportFormat, generate_report

# Lazy imports for heavy modules
# from organizer.ai import (...)
# from organizer.organizer import (...)

logger = logging.getLogger(__name__)

# Rich console for output
console = Console()


# =============================================================================
# Helper Functions
# =============================================================================


def print_header(text: str) -> None:
    """Print a styled header.

    Args:
        text: Header text.
    """
    console.print()
    console.print(Panel(text, style="bold blue", expand=False))
    console.print()


def print_success(text: str) -> None:
    """Print a success message with green checkmark.

    Args:
        text: Success message.
    """
    console.print(f"[green]✓[/green] {text}")


def print_warning(text: str) -> None:
    """Print a warning message with yellow icon.

    Args:
        text: Warning message.
    """
    console.print(f"[yellow]⚠[/yellow] {text}")


def print_error(text: str) -> None:
    """Print an error message with red icon.

    Args:
        text: Error message.
    """
    console.print(f"[red]✗[/red] {text}")


def print_info(text: str) -> None:
    """Print an info message.

    Args:
        text: Info message.
    """
    console.print(f"[blue]ℹ[/blue] {text}")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask for Y/N confirmation.

    Args:
        prompt: Confirmation prompt.
        default: Default value if user just presses Enter.

    Returns:
        True if confirmed.
    """
    return Confirm.ask(prompt, default=default)


def setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler.

    Args:
        verbose: Enable debug logging.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )


def get_version() -> str:
    """Get the package version."""
    try:
        return __version__
    except Exception:
        return "0.1.0"


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.version_option(version=get_version(), prog_name="Digital Life Narrative AI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str | None) -> None:
    """Digital Life Narrative AI - Reconstruct your life story from media exports.

    Use AI to weave your scattered photos and videos into a meaningful narrative.

    Quick start:
        organizer config set-key      # Set up your Gemini API key
        organizer analyze -i ~/Photos -o ./my_story

    For more information on a command:
        organizer COMMAND --help
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config

    setup_logging(verbose)


# =============================================================================
# Analyze Command (PRIMARY)
# =============================================================================


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_paths",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory to analyze (can specify multiple)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./life_story_report"),
    help="Output path for report (without extension)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["html", "json", "both"]),
    default="html",
    help="Output format",
)
@click.option(
    "--no-ai",
    is_flag=True,
    help="Skip AI analysis (statistics only)",
)
@click.option(
    "--privacy-mode",
    is_flag=True,
    help="Enable strict privacy (anonymize more data)",
)
@click.option(
    "--max-chapters",
    type=int,
    default=None,
    help="Override maximum chapters to detect",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    input_paths: tuple[Path, ...],
    output: Path,
    output_format: str,
    no_ai: bool,
    privacy_mode: bool,
    max_chapters: int | None,
) -> None:
    """Analyze media exports and generate your life story.

    This is the PRIMARY command. It:
    1. Detects what platforms your exports are from
    2. Parses all media with metadata
    3. Uses AI to identify life chapters and write narratives
    4. Generates a beautiful interactive report

    Example:
        organizer analyze -i ~/Downloads/takeout -i ~/Snapchat -o ./my_story
    """
    print_header("🧠 Digital Life Narrative AI")

    # Step 1: Detect sources
    console.print("[bold]Step 1:[/bold] Detecting export sources...")

    all_detections: list[DetectionResult] = []
    for path in input_paths:
        detections = detect_export_source(path)
        all_detections.extend(detections)
        if detections:
            for d in detections:
                confidence_str = d.confidence.value
                platform_str = d.platform.value
                print_success(
                    f"Found {platform_str} export in {path.name} " f"({confidence_str} confidence)"
                )
        else:
            print_warning(f"No recognized export format in {path}")

    if not all_detections:
        print_error("No supported export formats detected!")
        console.print()
        console.print("Supported platforms:")
        for platform in SourcePlatform:
            if platform != SourcePlatform.UNKNOWN:
                console.print(f"  • {platform.value}")
        console.print()
        console.print("Tip: Make sure you've extracted any ZIP files from your exports.")
        ctx.exit(1)

    # Show detection summary
    console.print()
    table = Table(title="Detected Sources", show_header=True)
    table.add_column("Platform", style="cyan")
    table.add_column("Path")
    table.add_column("Confidence")

    for d in all_detections:
        conf_style = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
        }.get(d.confidence.value, "white")
        table.add_row(
            d.platform.value,
            str(d.root_path),
            f"[{conf_style}]{d.confidence.value}[/{conf_style}]",
        )

    console.print(table)
    console.print()

    if not confirm_action("Proceed with analysis?", default=True):
        print_info("Analysis cancelled.")
        ctx.exit(0)

    # Step 2: Parse sources
    console.print()
    console.print("[bold]Step 2:[/bold] Parsing media files...")

    all_items: list[MediaItem] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing...", total=len(all_detections))

        for detection in all_detections:
            progress.update(task, description=f"Parsing {detection.platform.value}...")

            try:
                # Get config for parsing
                get_config()

                # Parse this source
                results = parse_all_sources(
                    [detection.root_path],
                    config=AnalysisConfig(),
                )

                for res in results:
                    all_items.extend(res.items)

                    if res.parse_errors:
                        for error in res.parse_errors[:3]:
                            print_warning(f"  {error}")

            except Exception as e:
                # Escape path for Rich markup (paths with [...] are interpreted as tags)
                escaped_path = str(detection.root_path).replace("[", "\\[").replace("]", "\\]")
                print_error(f"Failed to parse {escaped_path}: {e}")

            progress.advance(task)

    console.print()
    print_success(f"Parsed {len(all_items):,} media items")

    if not all_items:
        print_error("No media items found!")
        ctx.exit(1)

    # Show parsing summary
    _show_parsing_summary(all_items)

    # Step 3: AI Analysis
    console.print()
    console.print("[bold]Step 3:[/bold] AI Life Story Analysis...")

    # Prepare settings
    privacy_settings = PrivacySettings(
        anonymize_paths=True,
        local_only_mode=no_ai,
    )
    if privacy_mode:
        privacy_settings.truncate_captions = 50
        privacy_settings.hash_people_names = True

    analysis_config = AnalysisConfig()
    if max_chapters:
        analysis_config.max_chapters = max_chapters

    # Run analysis
    report = _run_analysis(all_items, no_ai, privacy_settings, analysis_config)

    if report is None:
        ctx.exit(1)

    # Step 4: Generate report
    console.print()
    console.print("[bold]Step 4:[/bold] Generating report...")

    format_map = {
        "html": ReportFormat.HTML,
        "json": ReportFormat.JSON,
        "both": ReportFormat.BOTH,
    }
    report_format = format_map[output_format]

    try:
        output_paths = generate_report(report, output, report_format)
        console.print()
        print_success("Report generated!")
        for path in output_paths:
            console.print(f"  📄 {path}")

    except Exception as e:
        print_error(f"Failed to generate report: {e}")
        ctx.exit(1)

    # Offer to open report
    console.print()
    html_path = next((p for p in output_paths if p.suffix == ".html"), None)
    if html_path and confirm_action("Open report in browser?", default=True):
        webbrowser.open(html_path.as_uri())

    # Final summary
    console.print()
    if report.is_fallback_mode:
        print_warning("Report generated in fallback mode (statistics only)")
        console.print("  To unlock AI-powered narratives: [bold]organizer config set-key[/bold]")
    else:
        print_success(f"✨ Your life story is ready! {len(report.chapters)} chapters discovered.")


def _show_parsing_summary(items: list[MediaItem]) -> None:
    """Show a summary of parsed items."""
    # Platform breakdown
    by_platform: dict[str, int] = {}
    for item in items:
        platform = item.source_platform.value
        by_platform[platform] = by_platform.get(platform, 0) + 1

    table = Table(title="Parsing Summary", show_header=True)
    table.add_column("Platform")
    table.add_column("Items", justify="right")

    for platform, count in sorted(by_platform.items(), key=lambda x: -x[1]):
        table.add_row(platform, f"{count:,}")

    table.add_row("[bold]Total[/bold]", f"[bold]{len(items):,}[/bold]")

    console.print(table)


def _run_analysis(
    items: list[MediaItem],
    no_ai: bool,
    privacy: PrivacySettings,
    config: AnalysisConfig,
) -> Any:
    """Run AI or fallback analysis."""
    # Import here to avoid circular imports and speed up CLI startup
    from organizer.ai import (
        AINotAvailableError,
        FallbackAnalyzer,
        LifeStoryAnalyzer,
        check_api_key_configured,
    )
    from organizer.ai.client import APIKeyMissingError

    if no_ai:
        print_info("Running in statistics-only mode (--no-ai)")
        console.print()
        fallback = FallbackAnalyzer(config)
        return fallback.analyze(items)

    # Check if AI is available
    if not check_api_key_configured():
        console.print()
        print_warning("Gemini API key not configured!")
        console.print("  AI-powered life story analysis requires an API key.")
        console.print("  Run: [bold]organizer config set-key[/bold]")
        console.print()

        if confirm_action("Continue with statistics-only fallback?", default=True):
            fallback = FallbackAnalyzer(config)
            return fallback.analyze(items)
        else:
            return None

    # Run AI analysis
    console.print()
    console.print("🧠 [bold cyan]Analyzing your life story with AI...[/bold cyan]")
    console.print("  This may take a few minutes depending on collection size.")
    console.print()

    try:
        analyzer = LifeStoryAnalyzer(config=config, privacy=privacy)

        # Create progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting analysis...", total=100)

            def progress_callback(stage: str, percent: float) -> None:
                progress.update(task, description=stage, completed=percent)

            # Run async analysis
            report = asyncio.run(analyzer.analyze(items, progress_callback))

        console.print()
        print_success(f"✨ AI has identified {len(report.chapters)} chapters in your life!")

        return report

    except APIKeyMissingError:
        print_error("API key error. Please reconfigure with: organizer config set-key")
        return None

    except AINotAvailableError as e:
        print_warning(f"AI analysis failed: {e}")
        if confirm_action("Continue with statistics-only fallback?", default=True):
            fallback = FallbackAnalyzer(config)
            return fallback.analyze(items)
        return None

    except Exception as e:
        print_error(f"Analysis failed: {e}")
        logger.exception("Analysis error")
        if confirm_action("Continue with statistics-only fallback?", default=True):
            fallback = FallbackAnalyzer(config)
            return fallback.analyze(items)
        return None


# =============================================================================
# Config Command Group
# =============================================================================


@cli.group()
def config() -> None:
    """Manage configuration and API keys."""
    pass


@config.command("set-key")
def config_set_key() -> None:
    """Securely set your Gemini API key.

    The key will be stored according to your configured storage backend:
    - Environment variable (default)
    - System keyring (recommended for security)
    - Encrypted file

    Get your API key at: https://makersuite.google.com/app/apikey
    """
    print_header("🔑 Configure Gemini API Key")

    console.print("Your API key is required for AI-powered life story analysis.")
    console.print("Get one at: [link]https://makersuite.google.com/app/apikey[/link]")
    console.print()

    # Get API key (hidden input)
    api_key = Prompt.ask("Enter your Gemini API key", password=True)

    if not api_key:
        print_error("No API key provided.")
        return

    # Validate format (basic check)
    if len(api_key) < 20:
        print_warning("API key seems too short. Are you sure it's correct?")
        if not confirm_action("Continue anyway?"):
            return

    # Store the key
    try:
        app_config = get_config()
        manager = APIKeyManager(
            app_config.key_storage_backend,
            app_config.encrypted_key_file_path,
        )
        manager.store_key(api_key)

        console.print()
        print_success("API key stored successfully!")
        console.print(f"  Storage: {app_config.key_storage_backend.value}")

        # Verify it can be retrieved
        retrieved = manager.retrieve_key()
        if retrieved:
            print_success("Key verification passed.")
            
            # New: Offer immediate API validation
            console.print()
            if confirm_action("Would you like to verify the key with a test API call?", default=True):
                from organizer.ai import get_client, AIClientError
                
                with console.status("[bold cyan]Testing API connectivity..."):
                    try:
                        client = get_client(api_key=retrieved)
                        if client.is_available():
                            print_success("API key is VALID and functional! ✨")
                        else:
                            print_error("API key is configured but test call failed.")
                    except AIClientError as e:
                        print_error(f"API validation failed: {e}")
                    except Exception as e:
                        print_error(f"An unexpected error occurred: {e}")
        else:
            print_warning("Could not verify stored key.")

    except Exception as e:
        print_error(f"Failed to store API key: {e}")


@config.command("test-key")
def config_test_key() -> None:
    """Validate your currently stored API key with a test call.
    
    This performs a minimal API request to ensure the key is correctly
    configured and has the necessary permissions.
    """
    print_header("🧪 Testing API Key Validity")
    
    from organizer.ai import get_client, AIClientError
    
    try:
        with console.status("[bold cyan]Retrieving stored key..."):
            client = get_client()
        
        with console.status("[bold cyan]Making test API call..."):
            if client.is_available():
                print_success(f"API key is VALID! (Model: {client.model_name})")
            else:
                print_error("API key is configured but the test call failed (returned False).")
                
    except AIClientError as e:
        print_error(f"API check failed: {e}")
    except Exception as e:
        print_error(f"An unexpected error occurred during test: {e}")


@config.command("show")
def config_show() -> None:
    """Show current configuration (API key is redacted)."""
    print_header("⚙️ Current Configuration")

    try:
        app_config = get_config()

        table = Table(show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        # AI settings
        table.add_row("AI Model", app_config.ai.model_name)
        table.add_row("Max Tokens", str(app_config.ai.max_tokens))
        table.add_row("Temperature", str(app_config.ai.temperature))
        table.add_row("Timeout (seconds)", str(app_config.ai.timeout_seconds))

        # Key storage
        table.add_row("Key Storage", app_config.key_storage_backend.value)

        # Privacy
        table.add_row("Anonymize Paths", str(app_config.privacy.anonymize_paths))
        table.add_row("Local Only Mode", str(app_config.privacy.local_only_mode))

        console.print(table)

        # Check if key is configured
        console.print()
        try:
            manager = APIKeyManager(
                app_config.key_storage_backend,
                app_config.encrypted_key_file_path,
            )
            if manager.is_key_configured():
                print_success("API key is configured ✓")
            else:
                print_warning("API key not configured")
                console.print("  Run: [bold]organizer config set-key[/bold]")
        except Exception:
            print_warning("Could not check API key status")

    except Exception as e:
        print_error(f"Failed to load config: {e}")


@config.command("reset")
def config_reset() -> None:
    """Reset configuration to defaults."""
    if not confirm_action("Reset all configuration to defaults?"):
        print_info("Cancelled.")
        return

    try:
        config_path = AppConfig.get_default_config_path()
        if config_path.exists():
            config_path.unlink()
            print_success("Configuration reset to defaults.")
        else:
            print_info("No custom configuration found.")
    except Exception as e:
        print_error(f"Failed to reset config: {e}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a specific configuration value.

    Example:
        organizer config set ai.model_name gemini-1.5-flash
        organizer config set privacy.local_only_mode true
    """
    try:
        app_config = get_config()

        # Parse the key path
        parts = key.split(".")

        # Handle configuration sections (e.g., ai.model_name)
        if len(parts) == 2:
            section, setting = parts
            if hasattr(app_config, section):
                section_obj = getattr(app_config, section)
                if hasattr(section_obj, setting):
                    # Get current value to determine target type
                    current = getattr(section_obj, setting)
                    
                    # Convert value to target type
                    if isinstance(current, bool):
                        value = value.lower() in ("true", "1", "yes")
                    elif isinstance(current, int):
                        value = int(value)
                    elif isinstance(current, float):
                        value = float(value)
                    
                    setattr(section_obj, setting, value)
                else:
                    print_error(f"Unknown setting: {key}")
                    return
            else:
                print_error(f"Unknown section: {section}")
                return
        
        # Handle top-level configuration (e.g., key_storage_backend)
        elif len(parts) == 1:
            key = parts[0]
            if hasattr(app_config, key):
                current = getattr(app_config, key)
                
                # Handle Enum types
                if isinstance(current, Enum):
                    try:
                        enum_type = type(current)
                        value = enum_type(value)
                    except ValueError:
                        valid_values = ", ".join([e.value for e in type(current)])
                        print_error(f"Invalid value for {key}. Valid options: {valid_values}")
                        return
                elif isinstance(current, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                
                setattr(app_config, key, value)
            else:
                print_error(f"Unknown setting: {key}")
                return
        else:
            print_error(f"Invalid key format: {key}")
            return

        # Save config
        config_path = AppConfig.get_default_config_path()
        app_config.save_to_yaml(config_path)
        print_success(f"Set {key} = {value}")

    except Exception as e:
        print_error(f"Failed to set config: {e}")


# =============================================================================
# Scan Command
# =============================================================================


@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def scan(path: Path) -> None:
    """Scan a directory to detect export sources.

    This is a quick utility to see what exports are in a directory
    without running full analysis.

    Example:
        organizer scan ~/Downloads/takeout
    """
    print_header("🔍 Scanning Directory")

    console.print(f"Scanning: {path}")
    console.print()

    detections = detect_export_source(path)

    if not detections:
        print_warning("No recognized export formats found.")
        console.print()
        console.print("Supported platforms:")
        for platform in SourcePlatform:
            if platform != SourcePlatform.UNKNOWN:
                console.print(f"  • {platform.value}")
        return

    # Show detections
    table = Table(title=f"Detected in {path.name}", show_header=True)
    table.add_column("Platform", style="cyan")
    table.add_column("Confidence")
    table.add_column("Evidence")

    for d in detections:
        conf_style = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
        }.get(d.confidence.value, "white")

        evidence = ", ".join(d.evidence[:3])
        if len(d.evidence) > 3:
            evidence += f" (+{len(d.evidence) - 3} more)"

        table.add_row(
            d.platform.value,
            f"[{conf_style}]{d.confidence.value}[/{conf_style}]",
            evidence,
        )

    console.print(table)

    # Quick file count
    console.print()
    total_files = sum(1 for _ in path.rglob("*") if _.is_file())
    console.print(f"Total files: {total_files:,}")

    console.print()
    # Escape path to avoid Rich markup interpretation
    escaped_path = str(path).replace("[", "\\[").replace("]", "\\]")
    console.print(f"To analyze: [bold]organizer analyze -i {escaped_path} -o ./my_story[/bold]")


# =============================================================================
# Organize Command
# =============================================================================


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory with media files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for organized files",
)
@click.option(
    "--report",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Use existing report for chapter names",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["copy", "move", "symlink"]),
    default="copy",
    help="Organization mode",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview without executing",
)
def organize(
    input_path: Path,
    output: Path,
    report: Path | None,
    mode: str,
    no_confirm: bool,
    dry_run: bool,
) -> None:
    """Organize media files into chapter-based folders.

    If a report is provided, uses AI-detected chapters for folder names.
    Otherwise, organizes by year/month.

    Example:
        organizer organize -i ~/Photos -o ./organized --report ./my_story.json
    """
    from organizer.organizer import MediaOrganizer, OrganizeMode

    print_header("📁 Organize Media Files")

    # Load report if provided
    life_report = None
    if report:
        try:
            import json

            from organizer.models import LifeStoryReport

            report_data = json.loads(report.read_text(encoding="utf-8"))
            life_report = LifeStoryReport.model_validate(report_data)
            print_success(f"Loaded report with {len(life_report.chapters)} chapters")
        except Exception as e:
            print_error(f"Failed to load report: {e}")
            print_info("Organizing by year/month instead.")

    # Parse source directory
    console.print()
    console.print("Parsing source directory...")

    config = AnalysisConfig()
    result = parse_all_sources([input_path], config=config)
    items = result.items

    if not items:
        print_error("No media items found in source directory.")
        return

    print_success(f"Found {len(items):,} media files")

    # Create organizer
    mode_map = {
        "copy": OrganizeMode.COPY,
        "move": OrganizeMode.MOVE,
        "symlink": OrganizeMode.SYMLINK,
    }
    organize_mode = mode_map[mode]

    organizer = MediaOrganizer(organize_mode)

    # Create plan
    console.print()
    console.print("Creating organization plan...")
    plan = organizer.plan_organization(items, life_report, output)

    if not plan:
        print_warning("No files to organize.")
        return

    # Show preview
    console.print()
    preview = organizer.preview_plan(plan)
    console.print(preview)

    if dry_run:
        console.print()
        print_info("Dry run - no files were modified.")
        return

    # Confirm
    if not no_confirm:
        console.print()
        if not confirm_action("Proceed with organization?", default=False):
            print_info("Cancelled.")
            return

    # Execute
    console.print()
    console.print("Organizing files...")

    org_result = organizer.execute_plan(plan)

    console.print()
    print_success(f"Organized {org_result.organized:,} files")
    if org_result.skipped:
        print_warning(f"Skipped {org_result.skipped} files")
    if org_result.errors:
        print_error(f"Errors: {len(org_result.errors)}")

    if org_result.undo_log_path:
        console.print()
        console.print(f"Undo log: {org_result.undo_log_path}")


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print()
        print_info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        # Escape any Rich markup in the error message
        error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
        print_error(f"Unexpected error: {error_msg}")
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
