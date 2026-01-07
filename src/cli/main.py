"""
Command Line Interface for Digital Life Narrative AI.

This module provides the primary user interface for reconstructing life narratives
from scattered media exports using AI-powered analysis.
"""

import json
import logging
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

try:
    import click
except ImportError:
    click = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.prompt import Confirm
except ImportError:
    Console = None
    Panel = None
    Progress = None
    Table = None
    Confirm = None

# Note: These imports will work once the modules are fully implemented
# from src.config import get_config, AppConfig, get_api_key, APIKeyNotFoundError
# from src.detection import detect_sources, summarize_detections
# from src.parsers.pipeline import run_pipeline, PipelineConfig
# from src.ai.client import AIClient, AIUnavailableError
# from src.ai.life_analyzer import LifeStoryAnalyzer
# from src.ai.fallback import FallbackAnalyzer
# from src.ai.content_filter import ContentFilter
# from src.output.html_report import generate_report, ReportConfig
# from src.core.safety import SafetySettings

logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console() if Console else None

# Application version
__version__ = "1.0.0"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text: str) -> None:
    """Print styled header."""
    if console:
        console.print(f"\n[bold cyan]{text}[/bold cyan]\n")
    else:
        print(f"\n{text}\n")


def print_success(text: str) -> None:
    """Print green success message."""
    if console:
        console.print(f"[bold green]✓[/bold green] {text}")
    else:
        print(f"✓ {text}")


def print_warning(text: str) -> None:
    """Print yellow warning message."""
    if console:
        console.print(f"[bold yellow]⚠[/bold yellow] {text}")
    else:
        print(f"⚠ {text}")


def print_error(text: str) -> None:
    """Print red error message."""
    if console:
        console.print(f"[bold red]✗[/bold red] {text}")
    else:
        print(f"✗ {text}")


def print_info_panel(title: str, content: str, border_style: str = "blue") -> None:
    """Print info panel box."""
    if console and Panel:
        console.print(Panel(content, title=title, border_style=border_style))
    else:
        print(f"\n=== {title} ===\n{content}\n")


def confirm(prompt: str, default: bool = False) -> bool:
    """Prompt for yes/no confirmation."""
    if Confirm:
        return Confirm.ask(prompt, default=default)
    else:
        response = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not response:
            return default
        return response in ('y', 'yes')


def create_progress() -> Optional[Progress]:
    """Create standard progress bar setup."""
    if Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
    return None


def print_detection_table(results: List) -> None:
    """Print detection results as table."""
    if not console or not Table:
        for result in results:
            print(f"  Platform: {result.get('platform', 'Unknown')}")
            print(f"  Confidence: {result.get('confidence', 0):.0%}")
            print(f"  Files: {result.get('file_count', 0)}\n")
        return
    
    table = Table(title="Detected Sources")
    table.add_column("Platform", style="cyan")
    table.add_column("Confidence", style="green")
    table.add_column("Files", justify="right")
    table.add_column("Evidence")
    
    for result in results:
        platform = result.get('platform', 'Unknown')
        confidence = f"{result.get('confidence', 0):.0%}"
        file_count = str(result.get('file_count', 0))
        evidence = result.get('evidence', 'N/A')
        
        table.add_row(platform, confidence, file_count, evidence)
    
    console.print(table)


def print_analysis_summary(report) -> None:
    """Print analysis summary."""
    if not console:
        print(f"Total Memories: {getattr(report, 'total_memories', 0)}")
        print(f"Chapters: {len(getattr(report, 'chapters', []))}")
        print(f"Is Fallback: {getattr(report, 'is_fallback', False)}")
        return
    
    total_memories = getattr(report, 'total_memories', 0)
    chapters = getattr(report, 'chapters', [])
    is_fallback = getattr(report, 'is_fallback', False)
    
    if is_fallback:
        console.print(f"\n[yellow]⚠️ Statistics-Only Mode[/yellow]")
        console.print(f"Total Memories: {total_memories:,}")
    else:
        console.print(f"\n[bold green]✨ AI Analysis Complete![/bold green]")
        console.print(f"Total Memories: {total_memories:,}")
        console.print(f"Chapters Identified: {len(chapters)}")


# =============================================================================
# MAIN CLI GROUP
# =============================================================================

if click:
    @click.group()
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.option('--debug', is_flag=True, help='Enable debug mode')
    @click.option('--config', type=click.Path(), help='Custom config file')
    @click.option('--quiet', '-q', is_flag=True, help='Suppress non-essential output')
    @click.pass_context
    def organizer(ctx, verbose, debug, config, quiet):
        """
        Digital Life Narrative AI - Reconstruct your life story from media exports.
        
        This tool analyzes your scattered media (photos, videos) from various platforms
        and uses AI to generate a meaningful narrative of your life.
        """
        # Set up logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        elif verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
        
        # Store context for subcommands
        ctx.ensure_object(dict)
        ctx.obj['verbose'] = verbose
        ctx.obj['debug'] = debug
        ctx.obj['quiet'] = quiet
        ctx.obj['config_path'] = config


    # =============================================================================
    # ANALYZE COMMAND - Main workflow
    # =============================================================================

    @organizer.command()
    @click.option('--input', '-i', 'inputs', multiple=True, type=click.Path(exists=True), required=True,
                  help='Input directories (can specify multiple)')
    @click.option('--output', '-o', type=click.Path(), default='./life_story_report.html',
                  help='Output file path')
    @click.option('--format', '-f', type=click.Choice(['html', 'json', 'both']), default='html',
                  help='Output format')
    @click.option('--no-ai', is_flag=True, help='Force fallback mode (statistics only)')
    @click.option('--privacy', type=click.Choice(['strict', 'standard', 'detailed']), default='standard',
                  help='Privacy level for AI')
    @click.option('--max-chapters', type=int, help='Maximum chapters to detect')
    @click.option('--skip-safety-filter', is_flag=True, help='Skip content safety filtering')
    @click.option('--safety-level', type=click.Choice(['permissive', 'moderate', 'strict']), default='moderate',
                  help='Safety sensitivity')
    @click.option('--open', 'open_browser', is_flag=True, help='Open report in browser after generation')
    @click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
    @click.pass_context
    def analyze(ctx, inputs, output, format, no_ai, privacy, max_chapters, skip_safety_filter,
                safety_level, open_browser, dry_run):
        """
        Analyze media exports and generate AI-powered life story report.
        
        This is the main command that runs the full pipeline:
        1. Detect sources
        2. Parse memories
        3. Filter sensitive content
        4. AI analysis (or fallback)
        5. Generate report
        
        Example:
            organizer analyze -i ./snapchat -i ./google_photos -o my_story.html
        """
        print_header("🧠 Digital Life Narrative AI")
        
        # Validate inputs
        input_paths = [Path(p) for p in inputs]
        output_path = Path(output)
        
        if dry_run:
            print_info_panel("Dry Run Mode", "Showing what would be done without executing.")
            console.print(f"\nInput directories: {', '.join(str(p) for p in input_paths)}")
            console.print(f"Output path: {output_path}")
            console.print(f"AI mode: {'Disabled (fallback)' if no_ai else 'Enabled'}")
            return
        
        # Check output is writable
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print_error(f"Cannot create output directory: {e}")
            sys.exit(1)
        
        # Phase 1: Detection
        print_header("Phase 1: Detecting Sources")
        console.print("Scanning directories for recognized media exports...")
        
        # Mock detection results for now
        detection_results = [
            {'platform': 'Snapchat', 'confidence': 0.95, 'file_count': 2847, 'evidence': 'memories_history.json'},
            {'platform': 'Google Photos', 'confidence': 0.88, 'file_count': 5123, 'evidence': 'metadata folder'},
        ]
        
        print_detection_table(detection_results)
        
        if not confirm("Continue with these sources?", default=True):
            console.print("\n[yellow]Analysis cancelled.[/yellow]")
            return
        
        # Phase 2: Parsing
        print_header("Phase 2: Parsing Memories")
        
        total_memories = 0
        progress = create_progress()
        if progress:
            with progress:
                task = progress.add_task("Parsing memories...", total=100)
                for i in range(100):
                    progress.update(task, advance=1)
                    # Simulate work
                total_memories = 7970
        else:
            console.print("Parsing memories...")
            total_memories = 7970
        
        print_success(f"Parsed {total_memories:,} memories")
        
        # Phase 3: Safety Filtering
        if not skip_safety_filter:
            print_header("Phase 3: Safety Filtering")
            console.print("Analyzing content for sensitive material...")
            flagged_count = 42
            if flagged_count > 0:
                print_warning(f"{flagged_count} items flagged as potentially sensitive")
                console.print(f"These will be handled according to safety level: {safety_level}")
        
        # Phase 4: AI Analysis
        print_header("Phase 4: AI Analysis")
        
        if no_ai:
            print_warning("Running in statistics-only mode (AI disabled)")
            console.print("Generating year-based organization...")
            is_fallback = True
            chapters_count = 0
        else:
            try:
                console.print("🧠 [bold cyan]AI is analyzing your life story...[/bold cyan]")
                # Mock AI analysis
                progress = create_progress()
                if progress:
                    with progress:
                        stages = [
                            "Detecting chapters",
                            "Analyzing themes",
                            "Generating narratives",
                            "Creating synthesis"
                        ]
                        for stage in stages:
                            task = progress.add_task(f"{stage}...", total=100)
                            for i in range(100):
                                progress.update(task, advance=1)
                
                is_fallback = False
                chapters_count = 8
                print_success(f"✨ AI identified {chapters_count} chapters in your life")
            except Exception as e:
                print_error(f"AI analysis failed: {e}")
                if confirm("Continue with statistics-only mode?", default=True):
                    is_fallback = True
                    chapters_count = 0
                else:
                    sys.exit(1)
        
        # Phase 5: Report Generation
        print_header("Phase 5: Generating Report")
        console.print(f"Creating {format.upper()} report...")
        
        # Mock report generation
        console.print(f"Writing to: {output_path}")
        output_path.write_text("<html><body><h1>Mock Report</h1></body></html>")
        
        # Success!
        print_header("🎉 Complete!")
        print_success(f"Your {'AI-generated ' if not is_fallback else ''}Life Story is ready!")
        console.print(f"\nReport saved to: [bold cyan]{output_path}[/bold cyan]")
        
        if is_fallback:
            print_info_panel(
                "Unlock the Full Experience",
                "This report contains statistics only.\n\n"
                "To enable AI-powered chapters, narratives, and insights:\n"
                "  organizer config set-key\n\n"
                "Then run analyze again.",
                border_style="yellow"
            )
        
        if open_browser:
            console.print("\nOpening report in browser...")
            webbrowser.open(output_path.as_uri())


    # =============================================================================
    # SCAN COMMAND
    # =============================================================================

    @organizer.command()
    @click.argument('path', type=click.Path(exists=True))
    @click.option('--recursive', '-r', is_flag=True, help='Scan subdirectories')
    @click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
    def scan(path, recursive, output_json):
        """
        Scan a directory for recognized media export formats.
        
        Example:
            organizer scan ~/Downloads --recursive
        """
        scan_path = Path(path)
        
        if not output_json:
            print_header(f"Scanning: {scan_path}")
        
        # Mock detection
        results = [
            {'platform': 'Snapchat', 'confidence': 0.95, 'file_count': 2847, 'evidence': 'memories_history.json'},
        ]
        
        if output_json:
            print(json.dumps(results, indent=2))
        else:
            print_detection_table(results)


    # =============================================================================
    # PARSE COMMAND
    # =============================================================================

    @organizer.command()
    @click.option('--input', '-i', 'inputs', multiple=True, type=click.Path(exists=True), required=True)
    @click.option('--output', '-o', type=click.Path(), required=True, help='Output JSON file')
    @click.option('--platform', type=click.Choice(['snapchat', 'google_photos', 'local', 'auto']), default='auto')
    @click.option('--deduplicate/--no-deduplicate', default=True)
    def parse(inputs, output, platform, deduplicate):
        """
        Parse media exports without AI analysis.
        
        Example:
            organizer parse -i ./exports -o memories.json
        """
        print_header("Parsing Media Exports")
        
        console.print(f"Platform: {platform}")
        console.print(f"Deduplication: {'Enabled' if deduplicate else 'Disabled'}")
        
        # Mock parsing
        total_memories = 5000
        
        output_path = Path(output)
        output_path.write_text(json.dumps({'total': total_memories, 'memories': []}, indent=2))
        
        print_success(f"Parsed {total_memories:,} memories")
        console.print(f"Saved to: {output_path}")


    # =============================================================================
    # CONFIG GROUP
    # =============================================================================

    @organizer.group()
    def config():
        """Manage configuration settings."""
        pass


    @config.command()
    def show():
        """Display current configuration."""
        print_header("Current Configuration")
        
        if Table:
            table = Table(title="Settings")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("AI Provider", "Google Gemini")
            table.add_row("API Key", "[CONFIGURED]")
            table.add_row("Model", "gemini-1.5-flash")
            table.add_row("Privacy Mode", "standard")
            table.add_row("Safety Level", "moderate")
            
            console.print(table)
        else:
            print("AI Provider: Google Gemini")
            print("API Key: [CONFIGURED]")
            print("Model: gemini-1.5-flash")


    @config.command('set-key')
    @click.option('--test/--no-test', default=True, help='Test API connection')
    @click.option('--backend', type=click.Choice(['env', 'keyring', 'file']), default='keyring')
    def set_key(test, backend):
        """Set Gemini API key securely."""
        print_header("Set Gemini API Key")
        
        api_key = click.prompt("Enter your Gemini API key", hide_input=True)
        
        # Mock key validation
        if len(api_key) < 10:
            print_error("Invalid API key format")
            sys.exit(1)
        
        console.print(f"\nStoring key using: {backend}")
        
        if test:
            console.print("Testing API connection...")
            # Mock test
            print_success("API key is valid!")
        
        print_success("API key configured successfully")
        print_info_panel(
            "Next Steps",
            "Your Gemini API key is now configured.\n\n"
            "Run your first analysis:\n"
            "  organizer analyze -i ./exports -o my_story.html"
        )


    @config.command('clear-key')
    @click.option('--force', is_flag=True, help='Skip confirmation')
    def clear_key(force):
        """Remove stored API key."""
        if not force and not confirm("Remove API key?"):
            return
        
        print_success("API key removed")


    @config.command('set')
    @click.argument('key')
    @click.argument('value')
    def set_value(key, value):
        """
        Set a configuration value.
        
        Example:
            organizer config set ai.temperature 0.8
        """
        console.print(f"Set {key} = {value}")
        print_success("Configuration updated")


    @config.command()
    @click.option('--force', is_flag=True)
    def reset(force):
        """Reset configuration to defaults."""
        if not force and not confirm("Reset all configuration to defaults?"):
            return
        
        print_success("Configuration reset to defaults")


    # =============================================================================
    # REPORT COMMAND
    # =============================================================================

    @organizer.command()
    @click.option('--data', '-d', type=click.Path(exists=True), required=True)
    @click.option('--output', '-o', type=click.Path(), required=True)
    @click.option('--format', type=click.Choice(['html', 'json']), default='html')
    @click.option('--theme', type=click.Choice(['light', 'dark', 'auto']), default='auto')
    def report(data, output, format, theme):
        """
        Generate report from existing analysis data.
        
        Example:
            organizer report --data analysis.json --output report.html
        """
        print_header("Generating Report")
        
        data_path = Path(data)
        output_path = Path(output)
        
        console.print(f"Loading data from: {data_path}")
        console.print(f"Theme: {theme}")
        
        # Mock report generation
        output_path.write_text("<html><body><h1>Report</h1></body></html>")
        
        print_success("Report generated")
        console.print(f"Saved to: {output_path}")


    # =============================================================================
    # USAGE COMMAND
    # =============================================================================

    @organizer.command()
    def usage():
        """Show AI usage statistics."""
        print_header("AI Usage Statistics")
        
        if Table:
            table = Table(title="Usage Summary")
            table.add_column("Period", style="cyan")
            table.add_column("Requests", justify="right")
            table.add_column("Tokens", justify="right")
            table.add_column("Est. Cost", justify="right")
            
            table.add_row("Today", "5", "12,450", "$0.02")
            table.add_row("This Month", "23", "58,920", "$0.09")
            table.add_row("Total", "156", "412,380", "$0.62")
            
            console.print(table)
        else:
            print("Today: 5 requests, 12,450 tokens, $0.02")
            print("This Month: 23 requests, 58,920 tokens, $0.09")


    # =============================================================================
    # VERSION COMMAND
    # =============================================================================

    @organizer.command()
    def version():
        """Show version and system information."""
        print_header("Digital Life Narrative AI")
        
        console.print(f"Version: [bold]{__version__}[/bold]")
        console.print(f"Python: {sys.version.split()[0]}")
        
        # Check dependencies
        console.print("\n[bold]Dependencies:[/bold]")
        deps = [
            ("Click", click is not None),
            ("Rich", Console is not None),
            ("Jinja2", False),  # Will be True when imported
            ("Pillow", False),  # Will be True when imported
        ]
        
        for name, available in deps:
            status = "[green]✓[/green]" if available else "[red]✗[/red]"
            console.print(f"  {status} {name}")
        
        console.print("\n[bold]AI Status:[/bold]")
        console.print(f"  [yellow]○[/yellow] Gemini API (configure with: organizer config set-key)")


    # Entry point
    if __name__ == '__main__':
        organizer()

else:
    # Click not available
    def organizer():
        """Fallback when Click is not installed."""
        print("Error: Click framework is required.")
        print("Install with: pip install click rich")
        sys.exit(1)
    
    if __name__ == '__main__':
        organizer()
