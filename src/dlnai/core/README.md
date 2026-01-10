# Core Memory Data Model

**Location:** `src/core/memory.py`

## Overview

The `Memory` class is the **foundational data model** for the Digital Life Narrative AI project. It represents a universal schema that ALL parsed media normalizes into, regardless of source platform (Snapchat, Google Photos, Facebook, local folders, etc.).

Think of a Memory as **a single moment captured in time** — a photo, video, message, check-in, or story. After parsing, you shouldn't be able to tell what platform a Memory came from except by checking its `source_platform` field.

## Design Principles

### 1. **Platform-Agnostic**

All parsers (Snapchat, Google Photos, Facebook, etc.) convert their platform-specific formats into the Memory schema. This creates a unified data layer.

### 2. **Privacy-by-Design**

The `to_ai_payload()` method is the **ONLY** way to prepare memories for AI consumption. It ensures:

- File paths are NEVER sent to AI
- GPS coordinates are NEVER sent (only place names)
- People names are anonymized
- Captions are truncated
- Original metadata is excluded

### 3. **Timezone-Aware**

ALL datetime fields MUST be timezone-aware. Naive datetimes are automatically converted to UTC with a warning.

### 4. **Confidence Tracking**

Every piece of data has an associated confidence level:

- **VERIFIED**: From explicit platform metadata
- **HIGH**: From reliable source (EXIF with GPS)
- **MEDIUM**: From parsing (filename patterns)
- **LOW**: From heuristics (folder names)
- **INFERRED**: AI-generated or guessed

### 5. **Deduplication Support**

Three strategies for identifying duplicates:

- **Content Hash**: Exact duplicates (same file in multiple places)
- **Metadata Hash**: Fuzzy duplicates (same moment, different device)
- **is_same_moment()**: Cross-platform duplicates (within time tolerance)

## Core Components

### Enums

#### MediaType

```python
MediaType.PHOTO           # Static images
MediaType.VIDEO           # Video files
MediaType.SCREENSHOT      # Screenshots (distinct from photos)
MediaType.LIVE_PHOTO      # iOS live photos
MediaType.STORY           # Ephemeral stories
MediaType.MESSAGE         # Chat messages
MediaType.CHECK_IN        # Location check-ins
MediaType.AUDIO           # Audio recordings
MediaType.DOCUMENT        # Documents/PDFs
MediaType.UNKNOWN         # Fallback
```

#### SourcePlatform

```python
SourcePlatform.SNAPCHAT
SourcePlatform.GOOGLE_PHOTOS
SourcePlatform.FACEBOOK
SourcePlatform.INSTAGRAM
SourcePlatform.WHATSAPP
SourcePlatform.IMESSAGE
SourcePlatform.ONEDRIVE
SourcePlatform.DROPBOX
SourcePlatform.LOCAL        # DSLR, manual transfers
SourcePlatform.UNKNOWN
```

#### ConfidenceLevel

```python
ConfidenceLevel.VERIFIED    # Highest confidence
ConfidenceLevel.HIGH
ConfidenceLevel.MEDIUM
ConfidenceLevel.LOW
ConfidenceLevel.INFERRED    # Lowest confidence
```

### Supporting Models

#### GeoPoint

Represents GPS coordinates with validation:

```python
point = GeoPoint(latitude=40.7829, longitude=-73.9654)
distance = point.distance_km(other_point)  # Haversine distance
approximate = point.to_approximate(precision=2)  # Privacy protection
```

#### Location

Hierarchical location information:

```python
location = Location(
    coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
    place_name="Central Park",
    locality="Manhattan",
    region="New York",
    country="United States",
    country_code="US"
)

display = location.to_display_string()  # "Central Park, Manhattan, New York"
ai_summary = location.to_ai_summary()   # "New York, US" (privacy-conscious)
```

#### PersonTag

Person mentions/tags with anonymization:

```python
person = PersonTag(
    name="Alice Smith",
    platform_id="alice_s_123",
    confidence=ConfidenceLevel.VERIFIED
)

anonymous = person.to_anonymous()  # name becomes "person_a1b2c3d4"
```

### Main Model: Memory

#### Minimal Example

```python
memory = Memory(
    source_platform=SourcePlatform.GOOGLE_PHOTOS,
    media_type=MediaType.PHOTO
)
# ID is auto-generated
# created_at can be None
```

#### Full Example

```python
memory = Memory(
    source_platform=SourcePlatform.GOOGLE_PHOTOS,
    media_type=MediaType.PHOTO,
    created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
    created_at_confidence=ConfidenceLevel.VERIFIED,
    location=Location(
        coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
        place_name="Central Park"
    ),
    people=[PersonTag(name="Alice"), PersonTag(name="Bob")],
    caption="Summer afternoon",
    width=1920,
    height=1080,
    camera_make="Apple",
    camera_model="iPhone 12"
)
```

## Key Methods

### Privacy: to_ai_payload()

**CRITICAL**: This is the ONLY way to prepare memories for AI.

```python
# Strict: Minimal data (month/year only)
payload = memory.to_ai_payload(privacy_level="strict")

# Standard: Balanced (default)
payload = memory.to_ai_payload(privacy_level="standard")

# Detailed: Maximum context (still privacy-conscious)
payload = memory.to_ai_payload(privacy_level="detailed")
```

**Never Included** (at any privacy level):

- `source_path`
- Exact GPS coordinates
- `content_hash`
- `original_metadata`

### Deduplication: compute_content_hash()

```python
hash1 = memory.compute_content_hash(Path("photo.jpg"))
# Compare hashes to find exact duplicates
```

### Fuzzy Deduplication: compute_metadata_hash()

```python
hash1 = memory.compute_metadata_hash()
# Hash of: timestamp (to minute) + platform + media_type + approximate location
# Finds same moment from different devices
```

### Cross-Platform Deduplication: is_same_moment()

```python
snapchat_memory = Memory(...)
google_memory = Memory(...)

if snapchat_memory.is_same_moment(google_memory, tolerance_seconds=60):
    # Same photo auto-backed up to both platforms
    pass
```

### Merging: merge_with()

```python
phone_memory = Memory(
    created_at=datetime(...),
    created_at_confidence=ConfidenceLevel.VERIFIED,
    # No location
)

dslr_memory = Memory(
    created_at=datetime(...),
    created_at_confidence=ConfidenceLevel.INFERRED,
    location=Location(...)  # Has location
)

merged = phone_memory.merge_with(dslr_memory)
# Prefers: VERIFIED timestamp from phone
#          Location from DSLR (only source with location)
#          Merged people lists (deduplicated)
```

### Helper Methods

```python
year = memory.get_year()           # 2020
month = memory.get_month()         # 7
year_month = memory.get_year_month()  # "2020-07"

days = memory.days_since(date(2020, 1, 1))  # 196

timeline_point = memory.to_timeline_point()
# Minimal dict for timeline visualization
```

## Business Rules

### Timezone Handling

1. All `created_at` datetimes MUST be timezone-aware
2. Naive datetimes are auto-converted to UTC with warning
3. Store `timezone_name` when available for display

### Privacy Filtering

1. `source_path` stored internally, NEVER sent to AI
2. `to_ai_payload()` is the gatekeeper for all AI data
3. Default privacy level is "standard", not "detailed"

### Deduplication Strategy

1. **Content Hash**: Same file in GooglePhotos + Dropbox
2. **Metadata Hash**: Same moment on phone + DSLR
3. **is_same_moment()**: Snapchat save + Google Photos backup

### Confidence Cascade

When merging:

- VERIFIED > HIGH > MEDIUM > LOW > INFERRED
- Prefer non-None over None
- Merge people lists (deduplicate by normalized_name)

## Edge Cases

### No Timestamp

```python
memory = Memory(source_platform=SourcePlatform.LOCAL, created_at=None)
# Valid, but AI analysis will deprioritize
```

### Naive Datetime

```python
memory = Memory(
    source_platform=SourcePlatform.LOCAL,
    created_at=datetime(2020, 7, 15, 12, 30)  # No timezone
)
# Auto-converted to UTC
# Warning added: "Naive datetime converted to UTC"
```

### Old Timestamp

```python
memory = Memory(
    source_platform=SourcePlatform.LOCAL,
    created_at=datetime(1985, 1, 1, tzinfo=timezone.utc)
)
# Warning: "Timestamp before 1990, may be parse error"
```

### Coordinates at (0, 0)

```python
memory = Memory(
    source_platform=SourcePlatform.LOCAL,
    location=Location(coordinates=GeoPoint(latitude=0.0, longitude=0.0))
)
# Warning: "Coordinates near (0,0) - likely error"
```

### Empty Caption

```python
memory = Memory(source_platform=SourcePlatform.LOCAL, caption="   ")
# Normalized to: caption=None
```

## Usage in Project

### For Parser Developers

**Your parser must:**

1. Import the Memory model
2. Convert platform-specific data to Memory objects
3. Set confidence levels appropriately
4. Return a list of Memory objects

```python
from dlnai.core.models import Memory, MediaType, SourcePlatform, ConfidenceLevel, Location

def parse(export_path: Path) -> list[Memory]:
    memories = []
    
    # Parse platform-specific data
    for item in parse_export(export_path):
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=parse_timestamp(item),
            created_at_confidence=ConfidenceLevel.VERIFIED,  # From EXIF
            location=parse_location(item),
            # ... other fields
        )
        memories.append(memory)
    
    return memories
```

### For AI Analyzer Developers

**Never consume Memory objects directly:**

```python
# ❌ WRONG: Direct access to sensitive data
for memory in memories:
    send_to_ai(memory.source_path)  # NEVER DO THIS

# ✅ CORRECT: Use to_ai_payload()
for memory in memories:
    safe_payload = memory.to_ai_payload(privacy_level="standard")
    send_to_ai(safe_payload)
```

## Testing

Run the demonstration:

```bash
python demo_memory.py
```

Run the test suite:

```bash
pytest tests/test_memory.py -v
```

## Success Criteria

The Memory model is complete when:

- ✅ All enums are defined with documented values
- ✅ Memory model passes validation for valid inputs
- ✅ Memory model rejects invalid inputs with clear errors
- ✅ `to_ai_payload()` never exposes sensitive data at any privacy level
- ✅ Timezone handling is bulletproof (no naive datetimes escape)
- ✅ Comprehensive docstrings (Google style)
- ✅ Complete type hints (mypy-clean)

## Architecture Integration

```
┌─────────────────────────────────────────────────┐
│              Platform Parsers                    │
│  (Snapchat, Google Photos, Facebook, Local)      │
└─────────────────┬───────────────────────────────┘
                  │ normalize to
                  ▼
┌─────────────────────────────────────────────────┐
│            Memory (Universal Schema)             │ ← YOU ARE HERE
│  - Platform-agnostic                            │
│  - Privacy-by-design                            │
│  - Deduplication support                        │
└─────────────────┬───────────────────────────────┘
                  │ to_ai_payload()
                  ▼
┌─────────────────────────────────────────────────┐
│               AI Analyzer                        │
│  (Gemini API for narrative reconstruction)       │
└─────────────────────────────────────────────────┘
```

## Related Files

- **Implementation**: `src/core/memory.py` (1000+ lines)
- **Tests**: `tests/test_memory.py` (comprehensive test suite)
- **Demo**: `demo_memory.py` (interactive demonstration)
- **Package**: `src/core/__init__.py` (clean exports)

## Future Enhancements

Potential additions to the Memory model:

- [ ] Face embeddings for visual similarity
- [ ] Audio transcriptions
- [ ] OCR text from images
- [ ] Multi-language caption support
- [ ] Video scene detection timestamps
