# Security Policy

Your memories are personal. Your security is our priority.

---

## Table of Contents

- [Threat Model Overview](#threat-model-overview)
- [API Key Security](#api-key-security)
- [Logging Security](#logging-security)
- [Content Safety](#content-safety)
- [File System Security](#file-system-security)
- [Input Validation](#input-validation)
- [Third-Party Dependencies](#third-party-dependencies)
- [Vulnerability Reporting](#vulnerability-reporting)
- [Security Checklist for Users](#security-checklist-for-users)

---

## Threat Model Overview

### What This Application Is

Digital Life Narrative AI is a **local-first tool** for personal media analysis:

- Processes media exports on your local machine
- Optionally uses cloud AI (Google Gemini) for narrative analysis
- Generates self-contained HTML reports stored locally
- No server infrastructure, no database, no user accounts

### Primary Threats Considered

We've designed the application to protect against these security risks:

| Threat | Impact | Mitigation |
| ------ | ------ | ---------- |
| **API Key Leakage** | Unauthorized access to your Gemini account | Multi-layer key protection, redacting filters |
| **Sensitive Data in Logs** | Personal information exposed in log files | Comprehensive logging hygiene, pattern redaction |
| **Unintended Content Exposure** | Sensitive media in shared reports | Content safety framework with configurable actions |
| **Malicious Input** | Crafted exports causing crashes or exploits | Input validation, error handling, safe parsing |

### Out of Scope Threats

The following are **not** within the application's security boundary:

- **Physical Device Security**: We assume your computer is physically secure
- **Network-Level Attacks**: MITM attacks, DNS poisoning (rely on HTTPS/TLS)
- **AI Model Security**: Google's responsibility for Gemini API security
- **Operating System Vulnerabilities**: We rely on OS-level security primitives
- **Social Engineering**: Users sharing sensitive reports is outside our control

---

## API Key Security

Your Gemini API key grants access to Google's AI services under your account. Protecting it is critical.

### Storage Mechanisms

We support three secure storage backends, tried in priority order:

#### 1. Environment Variable

```bash
export GEMINI_API_KEY="your-key-here"
```

**Use case**: CI/CD, containers, temporary development  
**Security**: Standard practice, but visible in process lists and shell history

#### 2. System Keyring (Recommended)

```bash
organizer config set-key
# Stored in:
# - Windows: Credential Manager
# - macOS: Keychain
# - Linux: Secret Service (GNOME Keyring, KWallet)
```

**Use case**: Personal desktop use  
**Security**: OS-level encryption, never stored in plaintext

#### 3. Encrypted File (Fallback)

```bash
# Automatically created at ~/.config/organizer/api_key.enc
# Encrypted with Fernet (AES-128-CBC)
# Key derived from machine ID (hostname + home path hash)
```

**Use case**: Headless systems, environments without keyring support  
**Security**: Encrypted at rest, but security depends on machine access control

### Security Guarantees

The application implements multiple layers of defense to prevent API key exposure:

| Protection Layer | Implementation | Location |
| ---------------- | -------------- | -------- |
| **Never printed to console** | Keys wrapped in `SecretStr` type | `src/config.py` |
| **Never in log messages** | `RedactingFilter` on all loggers | `src/ai/client.py` |
| **Never in error messages** | Exception handlers strip keys | Throughout codebase |
| **Never in stack traces** | Pydantic `SecretStr` redacts automatically | `src/config.py` |
| **Never in plaintext config** | Only encrypted or keyring storage | `src/config.py` |

### Implementation Details

#### APIKeyManager

The `APIKeyManager` class (in `src/config.py`) controls all key access:

- **Priority Fallback**: ENV → KEYRING → ENCRYPTED_FILE
- **Secure Deletion**: Overwrites key memory on deletion
- **Permission Enforcement**: Sets restrictive file permissions (600) on encrypted files
- **Validation**: Checks key format before storage

#### RedactingFilter

All loggers use `RedactingFilter` (in `src/ai/client.py`) to scan and redact:

- Patterns like `api_key=`, `key=`, `token=`, `bearer`
- Strings matching Gemini key format (`AIza[a-zA-Z0-9_-]{30,}`)
- Generic long alphanumeric strings (35-50 chars) that might be secrets

### Key Management Recommendations

#### 1. Never Commit Keys to Git

Our `.gitignore` already includes:

```gitignore
.env
.env.local
*.key
credentials.enc
api_key.enc
```

**Always verify** before committing configuration files.

#### 2. Rotate Keys Periodically

```bash
# Visit Google AI Studio
# https://makersuite.google.com/app/apikey

# Delete old key, generate new one
# Update locally:
organizer config set-key
```

**Recommended frequency**: Every 90 days, or immediately if compromised

#### 3. Revoke Unused Keys

If you stop using the application:

```bash
# Remove local key
organizer config reset

# Revoke in Google AI Studio
# https://makersuite.google.com/app/apikey
```

#### 4. Use Different Keys for Dev/Prod

- Create separate API keys for development and production use
- Set budget limits in Google Cloud Console to prevent unexpected charges
- Label keys clearly in Google AI Studio

---

## Logging Security

Logs can inadvertently expose sensitive information. We implement strict logging hygiene.

### What IS Logged

The application logs operational information for debugging:

| Category | Examples | Purpose |
| -------- | -------- | ------- |
| **Operation names** | "Starting analysis", "Parsing Snapchat export" | Progress tracking |
| **Token counts** | "Used 1,234 tokens" | Usage monitoring |
| **Timing** | "Analysis completed in 45.2s" | Performance monitoring |
| **Error types** | "FileNotFoundError", "JSONDecodeError" | Debugging |
| **File counts** | "Processed 500 memories" | Progress tracking |
| **Platform names** | "Detected Snapchat, Google Photos" | Source identification |

### What is NEVER Logged

The application **never** logs sensitive data:

| Data Type | Why Not | Protection |
| --------- | ------- | ---------- |
| **API keys or tokens** | Credentials | `RedactingFilter` |
| **Full file paths** | Contains usernames, folder structure | Path truncation |
| **Caption or description text** | May contain personal information | Explicit exclusion |
| **People's names** | Personal identifiers | Explicit exclusion |
| **GPS coordinates** | Location privacy | Explicit exclusion |
| **Full AI prompts** | May contain aggregated personal data | Explicit exclusion |
| **Full AI responses** | Contains generated narratives | Explicit exclusion |

### Implementation

#### RedactingFilter Class

Located in `src/ai/client.py`, this filter:

1. **Scans all log messages** before they're written
2. **Applies regex patterns** to detect sensitive data
3. **Replaces matches** with `[REDACTED]`
4. **Processes both message and arguments**

Patterns detected:

```python
# Key-value patterns
api_key=AIzaSy...  →  api_key=[REDACTED]
token=abc123...    →  token=[REDACTED]

# Standalone Gemini keys
AIzaSyABC123...    →  [REDACTED]

# Generic long strings (potential secrets)
a1b2c3d4e5f6...    →  [REDACTED]
```

#### Automatic Application

The filter is automatically added to:

- Root logger (affects all modules)
- AI client logger (defense in depth)
- All module-level loggers in `src/ai/`

### Log File Locations

Logs are written to platform-specific directories:

| Platform | Default Location |
| -------- | ---------------- |
| **Linux** | `~/.local/state/organizer/logs/` |
| **macOS** | `~/Library/Logs/organizer/` |
| **Windows** | `%LOCALAPPDATA%\organizer\logs\` |

**User Responsibility**: Secure these directories with appropriate file permissions. The application sets restrictive permissions on creation (700 for directories, 600 for files).

---

## Content Safety

The content safety system helps prevent accidental exposure of sensitive content in reports.

### Purpose

- Identify potentially sensitive media before it appears in reports
- Give users control over what's included in shareable outputs
- Provide warnings for content that might be inappropriate to share

### How It Works

The system operates in two modes:

#### 1. Metadata-Only Mode

Uses filename patterns, captions, and folder structure to detect:

```python
# Example patterns
NUDITY:    "nsfw", "nude", "boudoir"
VIOLENCE:  "blood", "gore", "injury"
PRIVATE:   "confidential", "vault", "hidden"
SUBSTANCE: "drunk", "high", "wasted"
```

**Accuracy**: Moderate (false positives and negatives expected)  
**Privacy**: No network calls, completely local

#### 2. Pixel-Aware Mode (Opt-In)

Uses Gemini Vision API to analyze image thumbnails:

- **Requires explicit disclosure acknowledgment**
- Sends 256x256 JPEG thumbnails (not full images)
- Higher accuracy than metadata-only
- Users can opt-out at any time

### Categories Detected

| Category | Description | Example Triggers |
| -------- | ----------- | ---------------- |
| `NUDITY` | Unclothed human bodies | Filename: "beach_nude.jpg" |
| `SEXUAL` | Sexually explicit content | Caption: "nsfw content" |
| `VIOLENCE` | Graphic violence, injury | Folder: "accident_photos/" |
| `SELF_HARM` | Self-harm depictions | Vision: Detected injury |
| `SUBSTANCE` | Drug/alcohol content | Caption: "drunk party" |
| `HATE` | Hate symbols or content | Vision: Detected symbols |
| `DISTURBING` | Generally disturbing | Vision: Disturbing imagery |
| `PRIVATE` | Sensitive documents | Folder: "vault/", Vision: ID card |

### Actions Available

Users configure how to handle each category:

| Action | Behavior | Use Case |
| ------ | -------- | -------- |
| `ALLOW` | Normal display in report | Non-sensitive content |
| `FLAG_ONLY` | Display with ⚠️ warning badge | Mildly sensitive, user discretion |
| `BLUR_IN_REPORT` | Show blurred, click to reveal | Sensitive but relevant |
| `HIDE_FROM_REPORT` | Exclude from visual report | Very sensitive, keep private |
| `REQUIRE_REVIEW` | Hold for manual user approval | Uncertain classification |

### Configuration Example

```yaml
# config.yaml
safety:
  enabled: true
  use_pixel_analysis: false  # Opt-out of vision analysis
  
  # Per-category actions
  category_actions:
    NUDITY: HIDE_FROM_REPORT
    VIOLENCE: BLUR_IN_REPORT
    PRIVATE: HIDE_FROM_REPORT
    SUBSTANCE: FLAG_ONLY
```

### Limitations

> [!WARNING]
> The content safety system is **not a complete content moderation solution**.

- **Metadata-based detection** has false positives and negatives
- **Filename patterns** can be fooled or miss content
- **AI vision** is probabilistic, not perfect
- **Users should review reports** before sharing publicly

The system is designed to **assist**, not replace, user judgment.

---

## File System Security

The application interacts with your file system carefully to prevent data loss and unauthorized access.

### Read Operations

| Operation | Behavior | Safety |
| --------- | -------- | ------ |
| **Original files** | Read-only access | Never modified |
| **Export JSON** | Parsed, never written | Safe parsing |
| **EXIF data** | Extracted, not modified | Read-only |

**Guarantee**: Your original files are **never modified or deleted** during analysis.

### Write Operations

| Operation | Location | Purpose |
| --------- | -------- | ------- |
| **HTML reports** | User-specified output path | Generated reports |
| **JSON reports** | User-specified output path | Structured data |
| **Config files** | `~/.config/organizer/` | Settings persistence |
| **Cache files** | Platform cache directory | Analysis cache |
| **Log files** | Platform log directory | Debugging |

#### File Organization (Optional)

The `organizer organize` command can copy files into chapter-based folders:

- **Default mode**: Copy files (originals preserved)
- **Undo log**: Creates `undo.json` for reversibility
- **Never deletes**: Original files always remain

### Permissions

The application attempts to set restrictive permissions on sensitive files:

| File Type | Permissions | Meaning |
| --------- | ----------- | ------- |
| **Config directory** | `700` (drwx------) | Owner-only access |
| **Credential files** | `600` (-rw-------) | Owner read/write only |
| **Log files** | `600` (-rw-------) | Owner read/write only |

> [!NOTE]
> On Windows, permissions are set using `icacls` to restrict access to the current user.

### Path Handling

To prevent path traversal and information leakage:

- **Paths are never included in AI requests**
- **Paths are not logged** (only filenames or counts)
- **Reports don't contain source paths** (unless explicitly configured)
- **Symlinks are handled carefully**: Follow file symlinks, caution with directory symlinks

---

## Input Validation

The application validates all external input to prevent crashes and exploits.

### Export Files

| Input | Validation | Error Handling |
| ----- | ---------- | -------------- |
| **JSON files** | Parsed with error handling | Malformed files logged and skipped |
| **Invalid data** | Pydantic validation | Invalid entries logged and skipped |
| **Missing fields** | Optional fields with defaults | Graceful degradation |

**Result**: Malformed export files don't crash the application.

### File Paths

| Input | Validation | Protection |
| ----- | ---------- | ---------- |
| **User-provided paths** | Validated before access | Path existence checks |
| **Symlinks** | Resolved carefully | Follow files, caution with directories |
| **Path traversal** | Prevented | No `../` traversal in exports |

**Result**: No path traversal vulnerabilities.

### Configuration

| Input | Validation | Fallback |
| ----- | ---------- | -------- |
| **Config files** | Pydantic validation | Safe defaults on invalid config |
| **Invalid values** | Type checking | Warnings logged, defaults used |
| **Missing config** | Optional with defaults | Application still functions |

**Result**: Invalid configuration doesn't prevent application startup.

### AI Responses

| Input | Validation | Error Handling |
| ----- | ---------- | -------------- |
| **JSON responses** | Parsed with error handling | Retry or fallback mode |
| **Unexpected format** | Schema validation | Logged and handled gracefully |
| **Malicious content** | Sanitized before display | HTML escaping in reports |

**Result**: Unexpected AI responses don't crash the application or cause XSS.

---

## Third-Party Dependencies

We carefully manage dependencies to minimize security risks.

### Key Dependencies

| Dependency | Purpose | Security Notes |
| ---------- | ------- | -------------- |
| **google-generativeai** | Official Google SDK | Maintained by Google |
| **Pydantic** | Data validation | Industry-standard validation |
| **Click** | CLI framework | Widely used, well-maintained |
| **Pillow** | Image processing | Regular security updates |
| **Jinja2** | Template rendering | Auto-escaping enabled |
| **keyring** | Secure credential storage | OS-level security primitives |
| **cryptography** | Encryption (Fernet) | Industry-standard crypto |

### Security Practices

#### 1. Pinned Versions

We use Poetry to pin exact dependency versions:

```toml
[tool.poetry.dependencies]
python = "^3.10"
google-generativeai = "^0.3.0"
pydantic = "^2.5.0"
# ... etc
```

**Benefit**: Reproducible builds, controlled updates

#### 2. Regular Updates

**Recommendation**: Check for dependency updates regularly:

```bash
poetry update
poetry show --outdated
```

**Frequency**: Monthly, or immediately for security advisories

#### 3. Vulnerability Scanning

**Recommendation**: Use tools like `safety` or `pip-audit`:

```bash
pip install safety
safety check --json
```

#### 4. Minimal Dependencies

We avoid unnecessary dependencies to reduce attack surface:

- No web frameworks (local-only application)
- No database drivers (no database)
- No network libraries beyond official SDKs

### Known Vulnerabilities

**As of January 2026**: No known vulnerable dependencies.

**Monitoring**: We track security advisories for all dependencies via GitHub Dependabot.

---

## Vulnerability Reporting

If you discover a security vulnerability, please report it responsibly.

### How to Report

> [!IMPORTANT]
> **Do NOT open a public GitHub issue for security vulnerabilities.**

**Contact**: Email [georgehampton08@gmail.com](mailto:georgehampton08@gmail.com) with:

1. **Description**: Clear explanation of the vulnerability
2. **Reproduction Steps**: How to reproduce the issue
3. **Potential Impact**: What could an attacker do?
4. **Suggested Fix** (optional): If you have ideas

**Response Time**: We aim to respond within **48 hours**.

### What to Report

Please report these types of issues:

| Issue Type | Examples |
| ---------- | -------- |
| **API Key Exposure** | Keys logged, printed, or stored insecurely |
| **Sensitive Data Leakage** | Personal data in logs, reports, or cache |
| **Path Traversal** | Ability to read/write outside intended directories |
| **Injection Vulnerabilities** | Command injection, code injection |
| **Content Safety Bypasses** | Ways to expose sensitive content unintentionally |
| **Dependency Vulnerabilities** | Known CVEs in dependencies |

### What's NOT a Vulnerability

These are **not** security vulnerabilities:

| Issue | Why Not |
| ----- | ------- |
| **Metadata visible in reports** | By design — users control what's analyzed |
| **AI generating unexpected content** | AI limitation, not a security flaw |
| **User choosing permissive settings** | User choice, documented behavior |
| **Reports containing personal data** | Expected — users analyze their own data |
| **Local-only mode being "too limited"** | Feature limitation, not security issue |

### Disclosure Policy

- **Private disclosure**: We'll work with you privately to fix the issue
- **Coordinated disclosure**: We'll agree on a public disclosure timeline
- **Credit**: We'll credit you in release notes (unless you prefer anonymity)
- **No bug bounty**: This is an open-source project with no budget

---

## Security Checklist for Users

Follow these best practices to use the application securely.

### Before Using

- [ ] **Understand what data is sent to AI**  
  Review [PRIVACY.md](PRIVACY.md) to understand what metadata goes to Gemini

- [ ] **Review privacy and safety settings**  
  Check `organizer config show` and adjust settings for your needs

- [ ] **Secure your API key**  
  Use system keyring (`organizer config set-key`) for best security

- [ ] **Review your exports**  
  Understand what data is in your platform exports before analysis

### During Use

- [ ] **Use appropriate privacy mode for your data**  
  - Public data: Default settings OK
  - Sensitive data: `--privacy-mode`
  - Maximum privacy: `--no-ai` (local-only)

- [ ] **Review safety flags before sharing reports**  
  Check for ⚠️ warnings in generated reports

- [ ] **Keep software updated**  

  ```bash
  pip install --upgrade digital-life-narrative-ai
  ```

- [ ] **Monitor API usage**  
  Check token usage in reports to avoid unexpected costs

### After Use

- [ ] **Secure or delete generated reports**  
  Reports contain your personal narrative — treat them as sensitive

- [ ] **Consider rotating API keys**  
  Especially if you've shared reports or logs

- [ ] **Clear logs if they contain sensitive timing data**  

  ```bash
  # Logs location varies by platform
  rm -rf ~/.local/state/organizer/logs/  # Linux
  ```

- [ ] **Review what was analyzed**  
  Check the JSON report to see what data was processed

### For Maximum Privacy

- [ ] **Use local-only mode exclusively**  

  ```bash
  organizer analyze -i ~/exports -o ./report --no-ai
  ```

- [ ] **Run in an isolated environment**  
  - Use a container or VM
  - Network-isolated environment
  - Process exports offline

- [ ] **Disable caching**  

  ```yaml
  # config.yaml
  ai:
    cache_enabled: false
  ```

- [ ] **Review the source code**  
  This project is open source — audit it yourself

---

## Questions?

If you have security concerns or questions:

1. **Review the code**: This project is open source at [github.com/georgehampton08-rgb/digital-life-narrative-ai](https://github.com/georgehampton08-rgb/digital-life-narrative-ai)
2. **Open an issue**: For non-sensitive questions, use [GitHub Issues](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/issues)
3. **Email privately**: For security concerns, email [georgehampton08@gmail.com](mailto:georgehampton08@gmail.com)

---

**Last updated:** January 2026

**Version:** 1.0.0
