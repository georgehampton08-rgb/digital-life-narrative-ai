# Export Preparation Guide

This guide explains how to prepare your Google Photos and Snapchat exports for analysis.

## Quick Start (Your Current Setup)

### ✅ Snapchat - Ready to Analyze

Your Snapchat export is already extracted and ready:

**Location**: `D:\Snapchat Memories Backup Data`  
**Status**: ✅ Has `memories_history.json` - Parser will detect this

**Run analysis**:

```bash
organizer analyze --input "D:\Snapchat Memories Backup Data" --output snapchat_story
```

### ⚠️ Google Takeout - Needs Extraction

Your Google Takeout files are still in ZIP format:

**Location**: `C:\Users\georg\Downloads\takeout-*.zip` (6 parts)  
**Status**: ❌ Not yet extracted

**Extraction steps**:

## Google Photos (Takeout) Exports

### What You'll Receive

When you download your Google Photos data from [Google Takeout](https://takeout.google.com/), you'll receive:

- **Single archive**: `takeout-YYYYMMDDTHHMMSSZ.zip` (if small enough)
- **Multi-part archives**: `takeout-YYYYMMDDTHHMMSSZ-001.zip`, `-002.zip`, etc. (for large exports)

### Extraction Steps

#### Option 1: Windows Built-in (Recommended for Small Exports)

1. Place all ZIP parts in the same folder
2. Right-click the **first part** (`-001.zip`)
3. Select "Extract All..."
4. Choose destination folder

#### Option 2: 7-Zip (Recommended for Large/Multi-part Exports)

1. Install [7-Zip](https://www.7-zip.org/) if not already installed
2. Place all ZIP parts in the same folder
3. Right-click the **first part** (`-001.zip`)
4. Select "7-Zip → Extract Here" or "Extract to..."
5. 7-Zip will automatically handle all parts

#### Option 3: WinRAR

1. Place all ZIP parts in the same folder
2. Right-click the **first part**
3. Select "Extract Here" or "Extract to..."

### Expected Directory Structure

After extraction, you should have:

```
Takeout/
└── Google Photos/
    ├── Photos from 2019/
    │   ├── IMG_1234.jpg
    │   ├── IMG_1234.jpg.json     # Sidecar with metadata
    │   ├── IMG_5678.jpg
    │   └── IMG_5678.jpg.json
    ├── Photos from 2020/
    │   └── ...
    ├── Album - Vacation/
    │   ├── photo1.jpg
    │   ├── photo1.jpg.json
    │   └── ...
    └── print-subscriptions.json  # Ignored by parser
```

### What the Parser Extracts

From each `.json` sidecar file:

- **Timestamp**: `photoTakenTime.timestamp` (Unix epoch)
- **Location**: `geoData.latitude` and `geoData.longitude`
- **People**: `people[].name`
- **Description**: `description` or `title`
- **Album**: Inferred from folder name (e.g., "Album - Vacation")

### Running the Analysis

```bash
# Point to the extracted Takeout directory
organizer analyze --input "C:\path\to\Takeout" --output my_story
```

---

## Snapchat Exports

### What You'll Receive

Snapchat exports come as:

- `mydata-YYYYMMDD.zip`

### Extraction Steps

1. Right-click the ZIP file
2. Select "Extract All..." (Windows) or use 7-Zip/WinRAR
3. Extract to a folder (e.g., `mydata`)

### Expected Directory Structure

```
mydata/
├── json/
│   ├── memories_history.json    # Main memories metadata
│   ├── account.json
│   ├── location_history.json
│   └── ...
├── html/
│   └── ... (web viewer, not used by parser)
└── media/
    └── ... (some exports put media here)

OR (older format):

snapchat_export/
├── memories_history.json
├── memories/
│   ├── 2020-01-15.jpg
│   ├── 2020-02-20.mp4
│   └── ...
├── chat_history/
│   └── friend_alice/
│       ├── messages.json
│       ├── shared_photo.jpg
│       └── ...
└── snap_history/
    └── snap_history.json
```

### What the Parser Extracts

From `memories_history.json`:

- **Timestamp**: `Date` field (various formats supported)
- **Location**: `Location` field (lat,lon string)
- **Media Type**: `Media Type` field

From chat history:

- Messages with timestamps
- Shared media files

### Running the Analysis

```bash
# Point to the extracted mydata directory
organizer analyze --input "C:\path\to\mydata" --output snapchat_story
```

---

## Common Issues

### Issue: "No exports detected"

**Cause**: Pointing to the ZIP file instead of the extracted folder

**Solution**: Extract the ZIP first, then point to the **extracted folder** containing the subdirectories

### Issue: "Found 0 items"

**Possible Causes**:

1. **Wrong directory level**: Make sure you're pointing to the correct level
   - For Google: Point to `Takeout` folder (contains `Google Photos` subfolder)
   - For Snapchat: Point to the root export folder (contains `memories_history.json` or `json/` folder)

2. **Missing JSON sidecar files**: Google Photos requires `.json` files alongside media
   - Check if `.json` files are present next to your photos
   - These are automatically included in Takeout exports

3. **Empty export**: Verify the export actually contains data

### Issue: Multi-part archives won't extract

**Solution**:

- Ensure **all parts** are in the same folder
- Always extract from **part 001**, not other parts
- Use 7-Zip for best compatibility with multi-part archives

---

## Advanced: Combining Multiple Exports

You can analyze multiple export sources together:

```bash
organizer analyze \
  --input "C:\Takeout" \
  --input "C:\mydata" \
  --input "D:\Photography Photos" \
  --output combined_story
```

The application will:

1. Auto-detect each source type (Google Photos, Snapchat, or Local)
2. Parse all sources independently
3. Merge and deduplicate items
4. Create a unified narrative

---

## File Format Support

### Google Photos

- **Images**: JPG, PNG, HEIC, HEIF, GIF, WebP, BMP, TIFF, RAW, DNG
- **Videos**: MP4, MOV, AVI, MKV, WebM, M4V, 3GP

### Snapchat

- **Images**: JPG, PNG
- **Videos**: MP4, MOV

### Local Files

- All formats supported by Google Photos parser
- Plus additional camera RAW formats (NEF, CR2, ARW, etc.)

---

## Privacy & Security

✅ **All parsing happens locally** on your computer
✅ No files are uploaded to external servers
✅ Export files remain in your control
✅ AI analysis is optional (use `--no-ai` flag to disable)

---

## Next Steps

1. **Extract your exports** using the steps above
2. **Scan to verify detection**:

   ```bash
   organizer scan "C:\path\to\extracted\folder"
   ```

3. **Run full analysis**:

   ```bash
   organizer analyze --input "C:\path\to\folder" --output my_story
   ```

4. **View your narrative**: Open `my_story/index.html` in your browser
