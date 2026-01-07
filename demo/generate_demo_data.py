"""
Demo Data Generator for Digital Life Narrative AI

Generates synthetic media exports for demonstration purposes.
No real personal data — all content is fake/generated.

Usage:
    python -m demo.generate_demo_data --output ./demo_data
    python -m demo.generate_demo_data --output ./demo_data --seed 42

Creates:
    - Snapchat-like export structure
    - Google Photos Takeout structure
    - Local photos with EXIF data
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None
    print("Warning: PIL not installed. Images will be placeholder files.")

try:
    import piexif
except ImportError:
    piexif = None


# =============================================================================
# Constants
# =============================================================================

DEMO_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

SAMPLE_CAPTIONS = [
    "First day at new job",
    "Trip to the lake",
    "Family dinner",
    "New apartment",
    "Birthday celebration",
    "Weekend hike",
    "Coffee with friends",
    "Home office setup",
    "Garden project",
    "Movie night",
    "Beach day",
    "Holiday gathering",
    "Morning run",
    "New city exploring",
    "Graduation day",
    "Concert night",
    "Cooking experiment",
    "Road trip adventure",
]

SAMPLE_LOCATIONS = [
    ("City A", 40.7128, -74.0060),
    ("City B", 34.0522, -118.2437),
    ("City C", 41.8781, -87.6298),
    ("City D", 29.7604, -95.3698),
    None,  # Some without location
    None,
]

COLORS = [
    (66, 135, 245),  # Blue
    (245, 166, 66),  # Orange
    (66, 245, 135),  # Green
    (245, 66, 135),  # Pink
    (135, 66, 245),  # Purple
    (245, 245, 66),  # Yellow
    (66, 245, 245),  # Cyan
    (180, 180, 180),  # Gray
]


# =============================================================================
# Image Creation Helpers
# =============================================================================


def create_colored_image(path: Path, color: tuple, size: tuple = (100, 100)) -> None:
    """Create a small solid-color image.

    Args:
        path: Where to save the image.
        color: RGB tuple for the color.
        size: Image dimensions.
    """
    if Image is None:
        # Fallback: create minimal JPEG bytes
        path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")
        return

    img = Image.new("RGB", size, color)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".png":
        img.save(path, "PNG")
    else:
        img.save(path, "JPEG", quality=85)


def create_image_with_exif(
    path: Path, datetime_taken: datetime, color: tuple, size: tuple = (100, 100)
) -> None:
    """Create image with embedded EXIF DateTimeOriginal.

    Args:
        path: Where to save the image.
        datetime_taken: Datetime to embed in EXIF.
        color: RGB tuple for the color.
        size: Image dimensions.
    """
    if Image is None:
        create_colored_image(path, color, size)
        return

    img = Image.new("RGB", size, color)
    path.parent.mkdir(parents=True, exist_ok=True)

    if piexif is not None:
        # Embed EXIF data
        exif_dict = {
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: datetime_taken.strftime("%Y:%m:%d %H:%M:%S"),
                piexif.ExifIFD.DateTimeDigitized: datetime_taken.strftime("%Y:%m:%d %H:%M:%S"),
            }
        }
        exif_bytes = piexif.dump(exif_dict)
        img.save(path, "JPEG", quality=85, exif=exif_bytes)
    else:
        # Save without EXIF
        img.save(path, "JPEG", quality=85)


# =============================================================================
# Sidecar/JSON Creation Helpers
# =============================================================================


def create_google_sidecar(
    path: Path,
    taken_time: datetime,
    description: Optional[str] = None,
    location: Optional[tuple] = None,
) -> None:
    """Create Google Photos JSON sidecar file.

    Args:
        path: Path for the JSON sidecar.
        taken_time: When the photo was taken.
        description: Optional photo description.
        location: Optional (name, lat, lon) tuple.
    """
    timestamp = str(int(taken_time.timestamp()))

    sidecar = {
        "title": path.stem.replace(".json", ""),
        "photoTakenTime": {
            "timestamp": timestamp,
            "formatted": taken_time.strftime("%b %d, %Y, %I:%M:%S %p UTC"),
        },
    }

    if description:
        sidecar["description"] = description

    if location:
        name, lat, lon = location
        sidecar["geoData"] = {
            "latitude": lat,
            "longitude": lon,
            "altitude": 0.0,
            "latitudeSpan": 0.0,
            "longitudeSpan": 0.0,
        }
        sidecar["geoDataExif"] = {
            "latitude": lat,
            "longitude": lon,
            "altitude": 0.0,
            "latitudeSpan": 0.0,
            "longitudeSpan": 0.0,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def create_snapchat_memories_json(memories: list[dict], path: Path) -> None:
    """Create Snapchat memories_history.json file.

    Args:
        memories: List of memory dictionaries.
        path: Where to save the JSON.
    """
    data = {"Saved Media": memories}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# =============================================================================
# Utility Functions
# =============================================================================


def random_datetime_in_year(year: int) -> datetime:
    """Generate a random datetime within a given year.

    Args:
        year: The year for the datetime.

    Returns:
        Random datetime in that year.
    """
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))

    return start + timedelta(seconds=random_seconds)


# =============================================================================
# Platform-Specific Generators
# =============================================================================


def generate_snapchat_export(base_dir: Path) -> int:
    """Create fake Snapchat export structure.

    Args:
        base_dir: Base directory for demo data.

    Returns:
        Number of memories created.
    """
    export_dir = base_dir / "snapchat_export"
    memories_dir = export_dir / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)

    # Create account.json
    account = {
        "Username": "demo_user",
        "Display Name": "Demo User",
        "Creation Date": "2017-01-15 10:30:00 UTC",
    }
    (export_dir / "account.json").write_text(json.dumps(account, indent=2))

    # Generate memories
    num_memories = random.randint(5, 8)
    memories_list = []

    for i in range(num_memories):
        year = random.choice(DEMO_YEARS)
        dt = random_datetime_in_year(year)
        color = random.choice(COLORS)
        location = random.choice(SAMPLE_LOCATIONS)

        filename = f"memory_{i+1:03d}.jpg"
        image_path = memories_dir / filename

        create_colored_image(image_path, color)

        memory_entry = {
            "Date": dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Media Type": "Image",
            "Location": location[0] if location else None,
        }
        memories_list.append(memory_entry)

    # Create memories_history.json
    create_snapchat_memories_json(memories_list, export_dir / "memories_history.json")

    return num_memories


def generate_google_photos_export(base_dir: Path) -> int:
    """Create fake Google Photos Takeout structure.

    Args:
        base_dir: Base directory for demo data.

    Returns:
        Number of photos created.
    """
    takeout_dir = base_dir / "Takeout" / "Google Photos"
    takeout_dir.mkdir(parents=True, exist_ok=True)

    num_photos = random.randint(6, 10)
    photos_created = 0

    # Distribute across years
    for year in random.sample(DEMO_YEARS, min(4, len(DEMO_YEARS))):
        year_dir = takeout_dir / f"Photos from {year}"
        year_dir.mkdir(parents=True, exist_ok=True)

        # 1-3 photos per year
        photos_this_year = random.randint(1, 3)

        for j in range(photos_this_year):
            if photos_created >= num_photos:
                break

            dt = random_datetime_in_year(year)
            color = random.choice(COLORS)
            location = random.choice(SAMPLE_LOCATIONS)
            caption = random.choice(SAMPLE_CAPTIONS) if random.random() > 0.4 else None

            filename = f"IMG_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = year_dir / filename
            sidecar_path = year_dir / f"{filename}.json"

            create_colored_image(image_path, color)
            create_google_sidecar(sidecar_path, dt, caption, location)

            photos_created += 1

    return photos_created


def generate_local_photos(base_dir: Path) -> int:
    """Create local photos directory with EXIF data.

    Args:
        base_dir: Base directory for demo data.

    Returns:
        Number of images created.
    """
    local_dir = base_dir / "local_photos"
    local_dir.mkdir(parents=True, exist_ok=True)

    num_photos = random.randint(4, 6)

    for i in range(num_photos):
        year = random.choice(DEMO_YEARS)
        dt = random_datetime_in_year(year)
        color = random.choice(COLORS)

        if random.random() > 0.3:
            # Regular photo with EXIF
            filename = f"IMG_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            create_image_with_exif(local_dir / filename, dt, color)
        else:
            # Screenshot (no EXIF, PNG)
            filename = f"Screenshot_{dt.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            create_colored_image(local_dir / filename, color)

    return num_photos


# =============================================================================
# Main Generator Function
# =============================================================================


def generate_demo_data(output_dir: Path) -> dict:
    """Generate all demo data.

    Args:
        output_dir: Where to create the demo data.

    Returns:
        Dictionary with counts of created items.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📁 Creating Snapchat export...")
    snapchat_count = generate_snapchat_export(output_dir)
    print(f"   Created snapchat_export/ with {snapchat_count} memories")

    print("📁 Creating Google Photos Takeout...")
    google_count = generate_google_photos_export(output_dir)
    print(f"   Created Takeout/Google Photos/ with {google_count} photos")

    print("📁 Creating local photos...")
    local_count = generate_local_photos(output_dir)
    print(f"   Created local_photos/ with {local_count} images")

    total = snapchat_count + google_count + local_count

    return {
        "snapchat": snapchat_count,
        "google_photos": google_count,
        "local": local_count,
        "total": total,
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo data for Digital Life Narrative AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m demo.generate_demo_data --output ./demo_data
    python -m demo.generate_demo_data --output ./demo_data --seed 42
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./demo_data",
        help="Output directory for demo data (default: ./demo_data)",
    )

    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output)

    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")

    print("🎬 Generating demo data for Digital Life Narrative AI...\n")

    counts = generate_demo_data(output_dir)

    print(f"\n✅ Demo data created in: {output_dir}")
    print(
        f"   Total: {counts['total']} synthetic memories spanning {DEMO_YEARS[0]}-{DEMO_YEARS[-1]}"
    )

    print("\n📌 Next steps:")
    print(f"   1. Run fallback analysis (no API key needed):")
    print(f"      organizer analyze --input {output_dir} --output demo_report.html --no-ai")
    print(f"\n   2. Or configure API key for full AI experience:")
    print(f"      organizer config set-key")
    print(f"      organizer analyze --input {output_dir} --output demo_report.html")


if __name__ == "__main__":
    main()
