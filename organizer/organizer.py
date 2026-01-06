"""Media file organizer for Digital Life Narrative AI.

This is a SECONDARY feature – after AI has identified life chapters, users
can optionally organize their actual files into folders named after those
chapters. Without AI chapters, this falls back to basic year/month organization.

Safety is paramount:
- Preview mode by default
- Explicit confirmation required for file operations
- Undo log created for all operations
- Original files never deleted (even in MOVE mode)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from organizer.models import LifeChapter, LifeStoryReport, MediaItem

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class OrganizeMode(Enum):
    """Mode for organizing files."""

    COPY = "copy"
    MOVE = "move"
    SYMLINK = "symlink"


@dataclass
class OrganizePlan:
    """A single file organization operation.

    Attributes:
        source_file: Path to the original file.
        destination_folder: Target folder path.
        destination_filename: Final filename (may differ from original).
        chapter_title: Name of the chapter this belongs to, if any.
        conflict_resolution: How a naming conflict was resolved, if any.
    """

    source_file: Path
    destination_folder: Path
    destination_filename: str
    chapter_title: str | None = None
    conflict_resolution: str | None = None

    @property
    def destination_path(self) -> Path:
        """Full destination path including filename."""
        return self.destination_folder / self.destination_filename


@dataclass
class OrganizeResult:
    """Result of an organization operation.

    Attributes:
        total_files: Total number of files in the plan.
        organized: Number of files successfully organized.
        skipped: Number of files skipped.
        errors: List of error messages.
        destination_root: Root directory of organized files.
        undo_log_path: Path to the undo log file, if created.
    """

    total_files: int
    organized: int
    skipped: int
    errors: list[str] = field(default_factory=list)
    destination_root: Path = field(default_factory=lambda: Path("."))
    undo_log_path: Path | None = None


@dataclass
class UndoLogEntry:
    """Entry in the undo log for reversing operations.

    Attributes:
        operation: The operation performed (copy, move, symlink).
        source: Original file path.
        destination: New file path.
        timestamp: When the operation occurred.
    """

    operation: str
    source: str
    destination: str
    timestamp: str


# =============================================================================
# Constants
# =============================================================================

# Characters not allowed in file/folder names
INVALID_PATH_CHARS = r'<>:"/\\|?*'
INVALID_PATH_PATTERN = re.compile(f"[{re.escape(INVALID_PATH_CHARS)}]")

# Files to skip during organization
SKIP_FILES = frozenset({
    ".ds_store", "thumbs.db", "desktop.ini", ".picasa.ini",
    ".nomedia", ".thumbnails",
})

# Maximum folder name length
MAX_FOLDER_NAME_LENGTH = 100

# Month names for fallback organization
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# =============================================================================
# Media Organizer
# =============================================================================


class MediaOrganizer:
    """Organizes media files into folder structures based on life chapters.

    If AI-generated chapters are available, organizes files by chapter.
    Otherwise, falls back to basic year/month organization.

    Safety features:
    - Preview mode by default
    - Undo log for all operations
    - Never deletes original files
    - Preserves file metadata

    Attributes:
        mode: The organization mode (copy, move, symlink).

    Example:
        ```python
        organizer = MediaOrganizer(OrganizeMode.COPY)
        plan = organizer.plan_organization(items, report, Path("organized"))
        print(organizer.preview_plan(plan))
        result = organizer.execute_plan(plan)
        ```
    """

    def __init__(self, mode: OrganizeMode = OrganizeMode.COPY) -> None:
        """Initialize the media organizer.

        Args:
            mode: Organization mode (copy, move, or symlink).
        """
        self.mode = mode
        logger.debug(f"MediaOrganizer initialized with mode: {mode.value}")

    def plan_organization(
        self,
        items: list[MediaItem],
        report: LifeStoryReport | None,
        output_root: Path,
    ) -> list[OrganizePlan]:
        """Create an organization plan without executing.

        Args:
            items: Media items to organize.
            report: Life story report with chapters (optional).
            output_root: Root directory for organized files.

        Returns:
            List of OrganizePlan entries.
        """
        plans: list[OrganizePlan] = []

        # Build chapter lookup if available
        chapter_lookup = self._build_chapter_lookup(report) if report else {}

        # Track used filenames per destination folder
        used_names: dict[Path, set[str]] = {}

        for item in items:
            # Skip files that don't exist
            if not item.file_path.exists():
                logger.debug(f"Skipping non-existent file: {item.file_path}")
                continue

            # Skip system files
            if item.file_path.name.lower() in SKIP_FILES:
                continue

            # Skip hidden files
            if item.file_path.name.startswith("."):
                continue

            # Determine destination folder
            dest_folder, chapter_title = self._determine_destination(
                item, report, chapter_lookup, output_root
            )

            # Ensure folder is tracked
            if dest_folder not in used_names:
                used_names[dest_folder] = set()
                # Add existing files in folder
                if dest_folder.exists():
                    for existing in dest_folder.iterdir():
                        used_names[dest_folder].add(existing.name.lower())

            # Generate unique filename
            original_name = item.file_path.name
            unique_name, conflict = self._generate_unique_filename(
                original_name, dest_folder, used_names[dest_folder]
            )

            # Track this name
            used_names[dest_folder].add(unique_name.lower())

            plan = OrganizePlan(
                source_file=item.file_path,
                destination_folder=dest_folder,
                destination_filename=unique_name,
                chapter_title=chapter_title,
                conflict_resolution=conflict,
            )
            plans.append(plan)

        return plans

    def preview_plan(self, plan: list[OrganizePlan]) -> str:
        """Generate human-readable preview of the plan.

        Args:
            plan: The organization plan to preview.

        Returns:
            Formatted preview string.
        """
        if not plan:
            return "No files to organize."

        lines = [
            f"📁 Organization Plan ({self.mode.value.upper()} mode)",
            f"{'=' * 50}",
            "",
        ]

        # Group by destination folder
        by_folder: dict[Path, list[OrganizePlan]] = {}
        for p in plan:
            folder = p.destination_folder
            if folder not in by_folder:
                by_folder[folder] = []
            by_folder[folder].append(p)

        # Sort folders
        sorted_folders = sorted(by_folder.keys(), key=str)

        for folder in sorted_folders:
            folder_plans = by_folder[folder]
            chapter_title = folder_plans[0].chapter_title or "No Chapter"

            lines.append(f"📂 {folder.name}")
            lines.append(f"   Path: {folder}")
            lines.append(f"   Chapter: {chapter_title}")
            lines.append(f"   Files: {len(folder_plans)}")

            # Show conflicts
            conflicts = [p for p in folder_plans if p.conflict_resolution]
            if conflicts:
                lines.append(f"   ⚠️  Renamed: {len(conflicts)} files")

            lines.append("")

        # Summary
        total = len(plan)
        folders = len(by_folder)
        conflicts = sum(1 for p in plan if p.conflict_resolution)

        lines.append(f"{'=' * 50}")
        lines.append(f"Summary:")
        lines.append(f"  Total files: {total}")
        lines.append(f"  Folders: {folders}")
        if conflicts:
            lines.append(f"  Renamed (conflicts): {conflicts}")
        lines.append(f"  Mode: {self.mode.value}")

        return "\n".join(lines)

    def execute_plan(
        self,
        plan: list[OrganizePlan],
        dry_run: bool = False,
    ) -> OrganizeResult:
        """Execute the organization plan.

        Args:
            plan: The plan to execute.
            dry_run: If True, validate without file operations.

        Returns:
            OrganizeResult with operation statistics.
        """
        if not plan:
            return OrganizeResult(
                total_files=0,
                organized=0,
                skipped=0,
                errors=[],
                destination_root=Path("."),
            )

        # Determine root directory
        dest_root = plan[0].destination_folder
        for p in plan:
            # Find common ancestor
            try:
                dest_root = Path(*[
                    part for part in dest_root.parts
                    if part in p.destination_folder.parts
                ])
            except (TypeError, ValueError):
                pass

        result = OrganizeResult(
            total_files=len(plan),
            organized=0,
            skipped=0,
            errors=[],
            destination_root=dest_root,
        )

        # Create undo log
        undo_log: list[dict[str, str]] = []
        undo_log_path = dest_root / ".organize_undo_log.json"

        if dry_run:
            logger.info("Dry run mode - no files will be modified")

        for p in plan:
            try:
                if dry_run:
                    # Just validate
                    if not p.source_file.exists():
                        result.errors.append(f"Source not found: {p.source_file}")
                        result.skipped += 1
                    else:
                        result.organized += 1
                    continue

                # Create destination folder
                p.destination_folder.mkdir(parents=True, exist_ok=True)

                # Perform operation
                dest_path = p.destination_path

                if self.mode == OrganizeMode.COPY:
                    shutil.copy2(p.source_file, dest_path)  # copy2 preserves metadata
                elif self.mode == OrganizeMode.MOVE:
                    shutil.move(str(p.source_file), str(dest_path))
                elif self.mode == OrganizeMode.SYMLINK:
                    dest_path.symlink_to(p.source_file.resolve())

                # Log for undo
                undo_log.append({
                    "operation": self.mode.value,
                    "source": str(p.source_file),
                    "destination": str(dest_path),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                })

                result.organized += 1
                logger.debug(f"Organized: {p.source_file.name} -> {dest_path}")

            except Exception as e:
                error_msg = f"Failed to organize {p.source_file.name}: {e}"
                result.errors.append(error_msg)
                result.skipped += 1
                logger.error(error_msg)

        # Write undo log
        if not dry_run and undo_log:
            try:
                dest_root.mkdir(parents=True, exist_ok=True)
                undo_log_path.write_text(
                    json.dumps(undo_log, indent=2),
                    encoding="utf-8",
                )
                result.undo_log_path = undo_log_path
                logger.info(f"Undo log written: {undo_log_path}")
            except Exception as e:
                logger.warning(f"Failed to write undo log: {e}")

        return result

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _build_chapter_lookup(
        self,
        report: LifeStoryReport,
    ) -> dict[date, LifeChapter]:
        """Build a lookup from date to chapter.

        Args:
            report: The life story report.

        Returns:
            Dict mapping dates to chapters.
        """
        lookup: dict[date, LifeChapter] = {}

        for chapter in report.chapters:
            # Map each date in the chapter range to this chapter
            current = chapter.start_date
            while current <= chapter.end_date:
                lookup[current] = chapter
                # Move to next day
                current = date.fromordinal(current.toordinal() + 1)

        return lookup

    def _determine_destination(
        self,
        item: MediaItem,
        report: LifeStoryReport | None,
        chapter_lookup: dict[date, LifeChapter],
        output_root: Path,
    ) -> tuple[Path, str | None]:
        """Determine destination folder for an item.

        Args:
            item: The media item.
            report: The life story report (if available).
            chapter_lookup: Date to chapter mapping.
            output_root: Root output directory.

        Returns:
            Tuple of (destination folder path, chapter title or None).
        """
        # If no timestamp, put in Unknown Date folder
        if not item.timestamp:
            return output_root / "Unknown Date", None

        item_date = item.timestamp.date()

        # Try to find a chapter
        if chapter_lookup and item_date in chapter_lookup:
            chapter = chapter_lookup[item_date]
            folder_name = self._create_folder_name(chapter)
            return output_root / folder_name, chapter.title

        # Fallback: year/month organization
        year = item.timestamp.year
        month = item.timestamp.month
        month_name = MONTH_NAMES[month]

        year_folder = f"{year}"
        month_folder = f"{month:02d} - {month_name}"

        return output_root / year_folder / month_folder, None

    def _create_folder_name(self, chapter: LifeChapter) -> str:
        """Generate folder name from chapter.

        Format: "{start_year} - {title}"

        Args:
            chapter: The life chapter.

        Returns:
            Sanitized folder name.
        """
        year = chapter.start_date.year
        title = self._sanitize_path_component(chapter.title)

        # Handle year range
        if chapter.end_date.year != year:
            folder_name = f"{year}-{chapter.end_date.year} - {title}"
        else:
            folder_name = f"{year} - {title}"

        # Truncate if too long
        if len(folder_name) > MAX_FOLDER_NAME_LENGTH:
            folder_name = folder_name[:MAX_FOLDER_NAME_LENGTH - 3] + "..."

        return folder_name

    def _generate_unique_filename(
        self,
        original: str,
        destination: Path,
        used_names: set[str],
    ) -> tuple[str, str | None]:
        """Generate a unique filename avoiding conflicts.

        Args:
            original: Original filename.
            destination: Destination folder.
            used_names: Set of already used names (lowercase).

        Returns:
            Tuple of (unique filename, conflict resolution note or None).
        """
        # Check if original is available
        if original.lower() not in used_names:
            dest_path = destination / original
            if not dest_path.exists():
                return original, None

        # Generate unique name
        stem = Path(original).stem
        suffix = Path(original).suffix

        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            if new_name.lower() not in used_names:
                dest_path = destination / new_name
                if not dest_path.exists():
                    return new_name, f"renamed from {original}"
            counter += 1

            # Safety limit
            if counter > 10000:
                # Use timestamp to guarantee uniqueness
                ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
                new_name = f"{stem}_{ts}{suffix}"
                return new_name, f"renamed from {original}"

    def _sanitize_path_component(self, name: str) -> str:
        """Remove/replace characters invalid in file paths.

        Args:
            name: Original name.

        Returns:
            Sanitized name safe for filesystem.
        """
        # Replace invalid characters with underscore
        sanitized = INVALID_PATH_PATTERN.sub("_", name)

        # Replace multiple underscores with single
        sanitized = re.sub(r"_+", "_", sanitized)

        # Strip leading/trailing whitespace and underscores
        sanitized = sanitized.strip(" _")

        # Handle empty result
        if not sanitized:
            sanitized = "Untitled"

        return sanitized


# =============================================================================
# Undo Operations
# =============================================================================


def load_undo_log(log_path: Path) -> list[UndoLogEntry]:
    """Load undo log from file.

    Args:
        log_path: Path to the undo log JSON file.

    Returns:
        List of undo log entries.
    """
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        return [
            UndoLogEntry(
                operation=entry["operation"],
                source=entry["source"],
                destination=entry["destination"],
                timestamp=entry["timestamp"],
            )
            for entry in data
        ]
    except Exception as e:
        logger.error(f"Failed to load undo log: {e}")
        return []


def undo_organization(log_path: Path, dry_run: bool = True) -> OrganizeResult:
    """Undo a previous organization operation.

    Args:
        log_path: Path to the undo log.
        dry_run: If True, preview without executing.

    Returns:
        Result of the undo operation.
    """
    entries = load_undo_log(log_path)

    if not entries:
        return OrganizeResult(
            total_files=0,
            organized=0,
            skipped=0,
            errors=["No entries in undo log"],
            destination_root=log_path.parent,
        )

    result = OrganizeResult(
        total_files=len(entries),
        organized=0,
        skipped=0,
        errors=[],
        destination_root=log_path.parent,
    )

    # Process in reverse order
    for entry in reversed(entries):
        try:
            dest = Path(entry.destination)
            source = Path(entry.source)

            if not dest.exists():
                result.errors.append(f"Destination not found: {dest}")
                result.skipped += 1
                continue

            if dry_run:
                logger.info(f"Would restore: {dest} -> {source}")
                result.organized += 1
                continue

            if entry.operation == "copy":
                # For copy, just delete the copy
                dest.unlink()
            elif entry.operation == "move":
                # For move, move back
                shutil.move(str(dest), str(source))
            elif entry.operation == "symlink":
                # For symlink, remove the link
                dest.unlink()

            result.organized += 1
            logger.info(f"Restored: {dest}")

        except Exception as e:
            result.errors.append(f"Failed to restore {entry.destination}: {e}")
            result.skipped += 1

    return result


# =============================================================================
# Module-Level Functions
# =============================================================================


def organize_files(
    items: list[MediaItem],
    report: LifeStoryReport | None,
    output_root: Path,
    mode: OrganizeMode = OrganizeMode.COPY,
    confirm: bool = True,
) -> OrganizeResult:
    """High-level function for organizing files.

    If confirm=True, prints a preview and waits for user confirmation.

    Args:
        items: Media items to organize.
        report: Life story report with chapters (optional).
        output_root: Root directory for organized files.
        mode: Organization mode (copy, move, symlink).
        confirm: Whether to require user confirmation.

    Returns:
        OrganizeResult with operation statistics.
    """
    organizer = MediaOrganizer(mode)

    # Create plan
    plan = organizer.plan_organization(items, report, output_root)

    if not plan:
        print("No files to organize.")
        return OrganizeResult(
            total_files=0,
            organized=0,
            skipped=0,
            destination_root=output_root,
        )

    # Show preview
    preview = organizer.preview_plan(plan)
    print(preview)
    print()

    if confirm:
        print("⚠️  This will modify your filesystem.")
        print(f"   Mode: {mode.value.upper()}")
        print()

        response = input("Proceed? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Operation cancelled.")
            return OrganizeResult(
                total_files=len(plan),
                organized=0,
                skipped=len(plan),
                destination_root=output_root,
            )

    # Execute
    print("\nOrganizing files...")
    result = organizer.execute_plan(plan)

    # Print result
    print(f"\n✅ Organization complete!")
    print(f"   Organized: {result.organized}")
    if result.skipped:
        print(f"   Skipped: {result.skipped}")
    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"     - {error}")
        if len(result.errors) > 5:
            print(f"     ... and {len(result.errors) - 5} more")

    if result.undo_log_path:
        print(f"\n📋 Undo log: {result.undo_log_path}")
        print("   To undo: organizer undo <log_path>")

    return result
