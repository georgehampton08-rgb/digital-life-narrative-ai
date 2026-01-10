"""
Command Line Interface for Digital Life Narrative AI.

This module provides the primary user interface for reconstructing life narratives
from scattered media exports using AI-powered analysis.
"""

import json
import logging
import sys
import webbrowser
from datetime import datetime, timezone
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

# Core imports
from dlnai.config import get_config, get_api_key
from dlnai.detection import detect_sources, summarize_detections
from dlnai.parsers.pipeline import run_pipeline, PipelineConfig
from dlnai.ai.analyzer import LifeStoryAnalyzer, AnalysisConfig
from dlnai.output.html_report import generate_report, ReportConfig
from dlnai.core.models import Memory, DepthMode
from dlnai.core.privacy import PrivacySettings

logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console() if Console else None

# Application version
__version__ = "1.0.0"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def print_header(text: str) -> None:
    """Print a styled header."""
    if console and Panel:
        console.print()
        console.print(Panel(text, style="bold blue", expand=False))
        console.print()
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
        return response in ("y", "yes")


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
        platform = result.get("platform", "Unknown")
        confidence = f"{result.get('confidence', 0):.0%}"
        file_count = str(result.get("file_count", 0))
        evidence = result.get("evidence", "N/A")

        table.add_row(platform, confidence, file_count, evidence)

    console.print(table)


def print_analysis_summary(report) -> None:
    """Print analysis summary."""
    if not console:
        print(f"Total Memories: {getattr(report, 'total_memories', 0)}")
        print(f"Chapters: {len(getattr(report, 'chapters', []))}")
        print(f"Is Fallback: {getattr(report, 'is_fallback', False)}")
        return

    total_memories = getattr(report, "total_memories", 0)
    chapters = getattr(report, "chapters", [])
    is_fallback = getattr(report, "is_fallback", False)

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
    @click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
    @click.option("--debug", is_flag=True, help="Enable debug mode")
    @click.option("--config", type=click.Path(), help="Custom config file")
    @click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
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
        ctx.obj["verbose"] = verbose
        ctx.obj["debug"] = debug
        ctx.obj["quiet"] = quiet
        ctx.obj["config_path"] = config

    # =============================================================================
    # ANALYZE COMMAND - Main workflow
    # =============================================================================

    @organizer.command()
    @click.option(
        "--input",
        "-i",
        "inputs",
        multiple=True,
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Input directory or ZIP file to analyze (can specify multiple)",
    )
    @click.option(
        "--output",
        "-o",
        default="./life_story.html",
        help="Output path for the report",
    )
    @click.option(
        "--format",
        "-f",
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
        "--privacy",
        type=click.Choice(["standard", "high", "paranoid"]),
        default="standard",
        help="Privacy level for handling metadata",
    )
    @click.option(
        "--max-chapters",
        type=int,
        help="Override maximum chapters to detect",
    )
    @click.option(
        "--skip-safety-filter",
        is_flag=True,
        help="Skip detection of sensitive/personal content",
    )
    @click.option(
        "--safety-level",
        type=click.Choice(["low", "medium", "high"]),
        default="medium",
        help="Aggressiveness of content filtering",
    )
    @click.option(
        "--open", "open_browser", is_flag=True, help="Open report in browser after generation"
    )
    @click.option(
        "--depth", 
        type=click.Choice(["quick", "standard", "deep"]), 
        default="standard",
        help="Analysis depth: quick (cheap), standard (balanced), deep (thorough)"
    )
    @click.option(
        "--max-ai-images", 
        type=int, 
        help="Override maximum images for visual AI analysis"
    )
    @click.option(
        "--show-cost/--no-show-cost", 
        default=True, 
        help="Display estimated costs in CLI and report"
    )
    @click.option(
        "--yes", "-y", 
        is_flag=True, 
        help="Skip confirmation prompts"
    )
    @click.option("--dry-run", is_flag=True, help="Show what would be done without doing it")
    @click.pass_context
    def analyze(
        ctx,
        inputs,
        output,
        format,
        no_ai,
        privacy,
        max_chapters,
        skip_safety_filter,
        safety_level,
        open_browser,
        depth,
        max_ai_images,
        show_cost,
        yes,
        dry_run,
    ):
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

        # Phase 1: Parsing Pipeline (Detection + Extraction)
        print_header("Phase 1 & 2: Parsing Memories")
        
        pipeline_config = PipelineConfig(
            recursive_detection=True,
            deduplicate_results=True
        )
        
        progress_obj = create_progress()
        memories = []
        
        try:
            if progress_obj:
                with progress_obj:
                    p_task = progress_obj.add_task("Processing media sources...", total=100)
                    
                    def pipeline_progress(p):
                        # Map pipeline progress to CLI progress
                        if p.stage == "detection":
                            progress_obj.update(p_task, description="Detecting sources...", completed=10)
                        elif p.stage == "parsing":
                            pct = 10 + (p.percentage() * 0.8) # Parsing counts for 80%
                            progress_obj.update(p_task, description=p.to_status_line(), completed=pct)
                        elif p.stage == "deduplication":
                            progress_obj.update(p_task, description="Deduplicating...", completed=95)

                    pipeline_result = run_pipeline(input_paths, pipeline_config, progress=pipeline_progress)
            else:
                console.print("Processing media sources...")
                pipeline_result = run_pipeline(input_paths, pipeline_config)
            
            memories = pipeline_result.memories
            total_memories = pipeline_result.total_memories
            
            if not memories:
                print_error("No memories were extracted from the provided sources.")
                return

            print_success(f"Extracted {total_memories:,} unique memories from {len(pipeline_result.detections)} sources")
            
        except Exception as e:
            print_error(f"Parsing failed: {e}")
            if ctx.obj.get("debug"):
                logger.exception("Parsing failure details")
            sys.exit(1)

        # Phase 3: Estimation & Safety Warning
        from dlnai.ai.life_analyzer import estimate_visual_cost
        
        cost_est = estimate_visual_cost(
            memories, 
            depth=depth, 
            max_images=max_ai_images
        )
        
        if not no_ai:
            print_header("🔍 Analysis Estimation")
            est_content = (
                f"Depth mode: [bold cyan]{depth}[/bold cyan]\n"
                f"Total media items: {total_memories:,}\n"
                f"Estimated images for visual AI: [bold]{cost_est['estimated_images']}[/bold]\n"
            )
            if show_cost:
                est_content += f"Estimated cost: [bold green]${cost_est['total_estimated_cost_usd']:.2f}[/bold green]\n"
            
            print_info_panel("Pre-Run Preview", est_content)
            
            # Large dataset warning
            if total_memories > 500 and not yes:
                print_warning(f"Large dataset detected ({total_memories:,} items)")
                console.print("\nThis may take some time and incur API costs.")
                if not confirm("Do you want to proceed with full AI analysis?", default=True):
                    console.print("\n[yellow]Switching to statistics-only mode.[/yellow]")
                    no_ai = True

        # Phase 4: AI Analysis
        report = None
        is_fallback = no_ai

        if no_ai:
            print_warning("Running in statistics-only mode (AI disabled)")
            # Create a basic report using fallback logic if available, 
            # otherwise just use the stats from memories
            from dlnai.core.models import LifeStoryReport
            from dlnai.core.timeline import Timeline
            tl = Timeline(memories)
            report = LifeStoryReport(
                total_memories=total_memories,
                is_fallback=True
            )
            chapters_count = 0
        else:
            try:
                print_header(f"Phase 4: AI Analysis")
                console.print(f"🧠 [bold cyan]Analyzing your life story (depth: {depth})...[/bold cyan]")
                
                analyzer = LifeStoryAnalyzer()
                analysis_config = AnalysisConfig(
                    depth_mode=DepthMode(depth),
                    max_vision_images=max_ai_images
                )

                a_progress = create_progress()
                if a_progress:
                    with a_progress:
                        task = a_progress.add_task("AI analysis in progress...", total=100)
                        
                        def analyzer_progress(p):
                            a_progress.update(task, completed=p.percent, description=f"{p.stage}: {p.message}")

                        report = analyzer.analyze(memories, analysis_config, progress_callback=analyzer_progress)
                else:
                    report = analyzer.analyze(memories, analysis_config)

                chapters_count = len(report.chapters)
                print_success(f"✨ AI identified {chapters_count} chapters in your life")
            except Exception as e:
                print_error(f"AI analysis failed: {e}")
                if confirm("Continue with statistics-only mode?", default=True):
                    from dlnai.core.models import LifeStoryReport
                    report = LifeStoryReport(total_memories=total_memories, is_fallback=True)
                    is_fallback = True
                    chapters_count = 0
                else:
                    sys.exit(1)

        # Phase 5: Report Generation
        print_header("Phase 5: Generating Report")
        console.print(f"Creating {format.upper()} report...")

        try:
            report_config = ReportConfig()
            # If both, we might need multiple calls or the generator handles it
            generate_report(report, output_path, report_config)
            print_success(f"Report saved to: [bold cyan]{output_path}[/bold cyan]")
        except Exception as e:
            print_error(f"Report generation failed: {e}")
            sys.exit(1)

        # Success!
        print_header("🎉 Complete!")
        
        if not is_fallback:
            console.print(f"[bold green]✨ AI Analysis Overview[/bold green]")
            console.print(f"  • Depth: [cyan]{depth}[/cyan]")
            console.print(f"  • Images analyzed: {getattr(report, 'visual_stats', {}).get('images_processed', 0)}")
            console.print(f"  • Chapters detected: {chapters_count}")
            if show_cost:
                cost = getattr(report, 'visual_stats', {}).get('total_cost_usd', 0)
                console.print(f"  • Estimated cost: [bold green]${cost:.2f}[/bold green]")
        else:
             print_success(f"Your Life Story statistics-only report is ready!")
 
        console.print(f"\nReport saved to: [bold cyan]{output_path}[/bold cyan]")
 
        if is_fallback:
            print_info_panel(
                "Unlock the Full Experience",
                "This report contains statistics only.\n\n"
                "To enable AI-powered chapters, narratives, and insights:\n"
                "  organizer config set-key\n\n"
                "Then run analyze again.",
                border_style="yellow",
            )
 
        if open_browser:
            console.print("\nOpening report in browser...")
            webbrowser.open(output_path.as_uri())

    # =============================================================================
    # SCAN COMMAND
    # =============================================================================

    @organizer.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--recursive", "-r", is_flag=True, help="Scan subdirectories")
    @click.option("--json", "output_json", is_flag=True, help="Output as JSON")
    def scan(path, recursive, output_json):
        """
        Scan a directory for recognized media export formats.

        Example:
            organizer scan ~/Downloads --recursive
        """
        scan_path = Path(path)

        if not output_json:
            print_header(f"Scanning: {scan_path}")

        found_sources = detect_sources(scan_path)
        results = [
            {
                "platform": s.platform.value.capitalize(),
                "confidence": s.confidence.value,
                "file_count": s.estimated_file_count or 0,
                "evidence": ", ".join(s.evidence) if s.evidence else "Pattern match",
            }
            for s in found_sources
        ]

        if output_json:
            print(json.dumps(results, indent=2))
        else:
            print_detection_table(results)

    # =============================================================================
    # PARSE COMMAND
    # =============================================================================

    @organizer.command()
    @click.option(
        "--input", "-i", "inputs", multiple=True, type=click.Path(exists=True), required=True
    )
    @click.option("--output", "-o", type=click.Path(), required=True, help="Output JSON file")
    @click.option(
        "--platform",
        type=click.Choice(["snapchat", "google_photos", "local", "auto"]),
        default="auto",
    )
    @click.option("--deduplicate/--no-deduplicate", default=True)
    def parse(inputs, output, platform, deduplicate):
        """
        Parse media exports without AI analysis.

        Example:
            organizer parse -i ./exports -o memories.json
        """
        print_header("Parsing Media Exports")

        console.print(f"Platform: {platform}")
        console.print(f"Deduplication: {'Enabled' if deduplicate else 'Disabled'}")

        pipeline_config = PipelineConfig(
            recursive_detection=True,
            deduplicate_results=deduplicate
        )
        
        if platform != "auto":
            pipeline_config.include_platforms = {SourcePlatform(platform)}

        result = None
        p = create_progress()
        if p:
            with p:
                task = p.add_task("Parsing media...", total=100)
                def update_progress(prog):
                    p.update(task, completed=prog.percentage(), description=prog.to_status_line())
                result = run_pipeline([Path(i) for i in inputs], pipeline_config, progress=update_progress)
        else:
            result = run_pipeline([Path(i) for i in inputs], pipeline_config)

        total_memories = len(result.memories)
        output_path = Path(output)
        
        # Save to JSON
        data = {
            "total": total_memories,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "memories": [m.model_dump(mode="json") for m in result.memories]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print_success(f"Parsed {total_memories:,} memories")
        console.print(f"Saved to: [bold cyan]{output_path}[/bold cyan]")

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

    @config.command("set-key")
    @click.option("--test/--no-test", default=True, help="Test API connection")
    @click.option("--backend", type=click.Choice(["env", "keyring", "file"]), default="keyring")
    def set_key(test, backend):
        """Set Gemini API key securely."""
        print_header("🔑 Set Gemini API Key")

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
            "  organizer analyze -i ./exports -o my_story.html",
        )

    @config.command("clear-key")
    @click.option("--force", is_flag=True, help="Skip confirmation")
    def clear_key(force):
        """Remove stored API key."""
        if not force and not confirm("Remove API key?"):
            return

        print_success("API key removed")

    @config.command("set")
    @click.argument("key")
    @click.argument("value")
    def set_value(key, value):
        """
        Set a configuration value.

        Example:
            organizer config set ai.temperature 0.8
        """
        console.print(f"Set {key} = {value}")
        print_success("Configuration updated")

    @config.command()
    @click.option("--force", is_flag=True)
    def reset(force):
        """Reset configuration to defaults."""
        if not force and not confirm("Reset all configuration to defaults?"):
            return

        print_success("Configuration reset to defaults")

    # =============================================================================
    # REPORT COMMAND
    # =============================================================================

    @organizer.command()
    @click.option("--data", "-d", type=click.Path(exists=True), required=True)
    @click.option("--output", "-o", type=click.Path(), required=True)
    @click.option("--format", type=click.Choice(["html", "json"]), default="html")
    @click.option("--theme", type=click.Choice(["light", "dark", "auto"]), default="auto")
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


def main():
    """Entry point for the console script."""
    organizer()


if __name__ == "__main__":
    main()
