# Privacy Policy & Data Handling

Your memories are personal. We take that seriously.

---

## Table of Contents

- [Our Commitment](#our-commitment)
- [What Data is Processed](#what-data-is-processed)
- [Privacy Settings](#privacy-settings)
- [API Key Security](#api-key-security)
- [Local-Only Mode](#local-only-mode)
- [Data Retention](#data-retention)
- [Recommendations](#recommendations)

---

## Our Commitment

### 🔒 Your Files Never Leave Your Computer

Digital Life Narrative AI processes your media **locally**. Your photos and videos are never uploaded anywhere. The actual image and video content stays on your machine — always.

### 📊 Only Metadata is Sent to AI

When you use AI analysis (with your Gemini API key), we send only:

- Dates (e.g., "June 2020")
- General locations (e.g., "Chicago" — not GPS coordinates)
- Platform names (e.g., "Snapchat", "Google Photos")
- Media types (e.g., "photo", "video")
- Statistical summaries (e.g., "45% of items are from 2020")

We **never** send your actual photos, videos, or personal file paths.

### 🎛️ You Control What Gets Analyzed

Every privacy setting is configurable. You can:

- Run in completely local mode (no AI, no network calls)
- Enable strict privacy mode to anonymize even more data
- Review exactly what will be analyzed before confirming

---

## What Data is Processed

### ✅ Processed Locally (Never Leaves Your Computer)

| Data Type | Purpose | Storage |
| --------- | ------- | ------- |
| **File paths** | Locate your media files | Memory only |
| **EXIF data** | Extract timestamps, camera info | Memory only |
| **Platform exports** | Parse JSON metadata files | Memory only |
| **GPS coordinates** | Convert to place names | Memory only |
| **Face tags** | Associate people with photos | Memory only |
| **Full captions** | Context extraction | Memory only |
| **File contents** | Hash for deduplication | Never stored |

### ☁️ Sent to Gemini AI (With Your API Key)

| Data Type | What's Sent | What's NOT Sent |
| --------- | ----------- | --------------- |
| **Timestamps** | "2020-06-15" | Full datetime with timezone |
| **Locations** | "Chicago" | GPS coordinates (41.8781, -87.6298) |
| **People** | Count only, or hashed names | Real names (in privacy mode) |
| **Captions** | First 100 characters | Full caption (in privacy mode) |
| **Platforms** | "snapchat" | Internal file structures |
| **Statistics** | "500 photos in 2020" | Individual file details |

### ❌ NEVER Sent Anywhere

| Data Type | Why Not |
| --------- | ------- |
| **Actual images/videos** | We only process metadata |
| **Full file paths** | Contains usernames, folder structure |
| **Raw GPS coordinates** | Could pinpoint your home |
| **Unredacted captions** | May contain sensitive info |
| **Email addresses** | Personal identifier |
| **Phone numbers** | Personal identifier |
| **Financial data** | Not relevant, never processed |

---

## Privacy Settings

Configure privacy via the config file or command-line flags.

### `--privacy-mode` Flag

Enables strict privacy filtering before any AI calls:

```bash
organizer analyze -i ~/exports -o ./report --privacy-mode
```

**Effects:**

- ✓ Captions truncated to 50 characters
- ✓ People names replaced with hashes
- ✓ Locations generalized (city only, no neighborhoods)
- ✓ File paths completely excluded

### Privacy Configuration Options

```yaml
# config.yaml
privacy:
  # Completely anonymize file paths (recommended: true)
  anonymize_paths: true
  
  # Never send any data externally
  local_only_mode: false
  
  # Maximum caption length sent to AI
  max_caption_length: 100
  
  # Hash people's names before sending
  hash_people_names: false
  
  # Exclude location data entirely (optional)
  exclude_locations: false
```

### Setting Explanations

| Setting | Default | Description |
| ------- | ------- | ----------- |
| `anonymize_paths` | `true` | Remove file paths from AI prompts |
| `local_only_mode` | `false` | Disable all AI/network calls |
| `max_caption_length` | `100` | Truncate long captions |
| `hash_people_names` | `false` | Replace "Alice Smith" with "p_a1b2c3" |
| `exclude_locations` | `false` | Remove all location data from AI prompts |

---

## API Key Security

Your Gemini API key is sensitive — it grants access to Google's AI services under your account.

### Storage Options

We support three secure storage backends:

#### 1. Environment Variable (Recommended for CI/CD)

```bash
export GEMINI_API_KEY="your-key-here"
```

**Pros**: Standard practice, works in containers, easy to rotate
**Cons**: Visible in process lists, shell history

#### 2. System Keyring (Recommended for Personal Use)

```bash
organizer config set-key
# Key stored in:
# - Windows: Credential Manager
# - macOS: Keychain
# - Linux: Secret Service (GNOME Keyring, KWallet)
```

**Pros**: OS-level encryption, never in plaintext
**Cons**: Requires desktop session on Linux

#### 3. Encrypted File (Fallback)

```bash
# Stored at ~/.config/organizer/api_key.enc
# Encrypted with Fernet (AES-128-CBC)
# Key derived from machine ID
```

**Pros**: Works everywhere
**Cons**: Security depends on machine access

### Key Management Recommendations

1. **Never commit keys to git**

   ```gitignore
   # Already in .gitignore
   .env
   .env.local
   *.key
   credentials.enc
   ```

2. **Rotate keys periodically**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Delete old key, generate new one
   - Update locally: `organizer config set-key`

3. **Delete keys when not in use**

   ```bash
   organizer config reset
   ```

4. **Use different keys for dev/prod**
   - Create separate API keys
   - Set budget limits in Google Cloud Console

---

## Local-Only Mode

For maximum privacy, run without any external network calls.

### Enable Local-Only Mode

```bash
# Via command line
organizer analyze -i ~/exports -o ./report --no-ai

# Via config
organizer config set privacy.local_only_mode true
```

### What You Get in Local-Only Mode

| Feature | Available | Notes |
| ------- | --------- | ----- |
| **Source detection** | ✅ Yes | Fully local |
| **Media parsing** | ✅ Yes | All metadata extracted |
| **Statistics** | ✅ Yes | Complete stats |
| **Year-based chapters** | ✅ Yes | Calendar organization |
| **AI narratives** | ❌ No | Requires Gemini |
| **Semantic chapters** | ❌ No | Requires Gemini |
| **Life story summary** | ❌ No | Requires Gemini |
| **Platform insights** | ❌ No | Requires Gemini |

### Network Activity

In local-only mode, Digital Life Narrative AI makes **zero network requests**:

```text
┌────────────────────────────────────────┐
│         Your Computer                   │
│                                         │
│   Exports ──► Parser ──► Report         │
│                                         │
│   ✓ No DNS lookups                     │
│   ✓ No HTTP requests                   │
│   ✓ No API calls                       │
│   ✓ Complete privacy                   │
│                                         │
└────────────────────────────────────────┘
```

---

## Data Retention

### Our Application

**We retain nothing.**

- No data is stored in any database
- No analytics or telemetry
- No crash reports sent
- No usage tracking
- Reports are saved only to your local filesystem

### Gemini API (Google)

When you use the AI features with your API key, Google's data policies apply:

- **API Data**: Google's [Gemini API Terms](https://ai.google.dev/terms) govern data handling
- **Retention**: As of 2024, Gemini API inputs are not used for training by default
- **Your Control**: You manage your API key and can revoke access anytime

**Important**: Review Google's current policies at:

- [Gemini API Terms of Service](https://ai.google.dev/terms)
- [Google Privacy Policy](https://policies.google.com/privacy)

### Generated Reports

Reports (`life_story.html`, `life_story.json`) are saved to your specified output path:

- Stored only on your local filesystem
- You control deletion and sharing
- No copies are made elsewhere

---

## Recommendations

### Before Analysis

1. **Review your exports**
   - Understand what platform exports contain
   - Remove any files you don't want analyzed
   - Check for sensitive captions or location data

2. **Start with local mode**

   ```bash
   organizer analyze -i ~/exports -o ./report --no-ai
   ```

   - Review what's detected before enabling AI
   - Check the JSON output for data overview

### During Analysis

1. **Use privacy mode for sensitive data**

   ```bash
   organizer analyze -i ~/exports -o ./report --privacy-mode
   ```

   - Hashes people names
   - Truncates captions
   - Generalizes locations

2. **Analyze in batches**
   - Process different time periods separately
   - Exclude directories with highly sensitive content

### After Analysis

1. **Review reports before sharing**
   - Open `life_story.html` in a browser
   - Check chapter narratives for sensitive details
   - AI may infer information from patterns — review these

2. **Sanitize shared reports**
   - Consider what narratives reveal about locations, habits
   - Remove or edit chapters before sharing publicly
   - The JSON file contains structured data — share cautiously

3. **Clean up**

   ```bash
   # Remove API key when not in use
   organizer config reset
   
   # Delete reports you no longer need
   rm ./life_story.html ./life_story.json
   ```

### For Maximum Privacy

1. **Use local-only mode exclusively**
   - Accept statistical-only analysis
   - No data ever leaves your machine

2. **Run in an isolated environment**
   - Use a container or VM
   - Network-isolated environment
   - Process exports offline

---

## Questions?

If you have privacy concerns or questions:

1. **Review the code**: This project is open source
2. **Open an issue**: [GitHub Issues](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/issues)
3. **Suggest improvements**: PRs welcome for privacy enhancements

---

**Last updated:** January 2026
