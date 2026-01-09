"""
HTML Report Generator - Transform LifeStoryReport into beautiful HTML.

This module generates self-contained, interactive HTML reports showcasing
AI-generated life narratives from user's media exports.
"""

import base64
import html as html_lib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    from jinja2 import Environment, Template
except ImportError:
    Environment = None
    Template = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Imports will be available after other modules are created
# from src.ai import LifeStoryReport, LifeChapter
# from src.core.memory import Memory, MediaType, SourcePlatform
# from src.core.chapter import LifeChapter, DateRange
# from src.core.safety import MemorySafetyState, SafetyAction, get_display_treatment, SafetySettings

logger = logging.getLogger(__name__)

# Emoji Mappings for Visual Context
SCENE_EMOJI = {
    "home": "🏠",
    "office": "🏢",
    "restaurant": "🍽️",
    "cafe": "☕",
    "nature": "🌲",
    "beach": "🏖️",
    "city": "🏙️",
    "urban": "🏙️",
    "indoor": "🏠",
    "outdoor": "🌲",
    "park": "🌳",
    "street": "🛣️",
    "transport": "🚗",
    "school": "🏫",
    "gym": "💪",
    "venue": "🎪",
    "concert": "🎸",
    "theater": "🎭",
    "museum": "🏛️",
    "store": "🛒",
    "market": "🛍️",
    "airport": "✈️",
    "station": "🚉",
    "waterfront": "🌊",
}

VIBE_EMOJI = {
    "adventure": "🧭",
    "social": "👥",
    "calm": "🧘",
    "reflective": "💭",
    "professional": "💼",
    "energetic": "⚡",
    "nostalgic": "🕰️",
    "creative": "🎨",
    "festive": "🎉",
    "intimate": "❤️",
    "cozy": "🧶",
    "modern": "✨",
    "rustic": "🪵",
    "official": "🏛️",
}

MOTIF_EMOJI = {
    "phone": "📱",
    "laptop": "💻",
    "coffee": "☕",
    "food": "🥘",
    "drink": "🍹",
    "pet": "🐾",
    "dog": "🐶",
    "cat": "🐱",
    "car": "🚗",
    "book": "📖",
    "music": "🎵",
    "art": "🖼️",
    "plant": "🌱",
    "sun": "☀️",
    "mountain": "⛰️",
    "water": "💧",
}


# =============================================================================
# EMBEDDED CSS
# =============================================================================

EMBEDDED_CSS = """
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #8b5cf6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-primary: #ffffff;
    --bg-secondary: #f9fafb;
    --bg-tertiary: #f3f4f6;
    --text-primary: #111827;
    --text-secondary: #6b7280;
    --text-tertiary: #9ca3af;
    --border: #e5e7eb;
    --shadow: rgba(0, 0, 0, 0.1);
    --radius: 12px;
    --spacing: 1rem;
}

[data-theme="dark"] {
    --bg-primary: #111827;
    --bg-secondary: #1f2937;
    --bg-tertiary: #374151;
    --text-primary: #f9fafb;
    --text-secondary: #d1d5db;
    --text-tertiary: #9ca3af;
    --border: #374151;
    --shadow: rgba(0, 0, 0, 0.3);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg-secondary);
    color: var(--text-primary);
    line-height: 1.6;
    transition: background 0.3s, color 0.3s;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
}

h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 2rem;
    font-weight: 600;
    margin: 2rem 0 1rem;
    color: var(--text-primary);
}

h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem;
}

.subtitle {
    color: var(--text-secondary);
    font-size: 1.125rem;
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
    margin-right: 0.5rem;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
}

.badge-primary {
    background: var(--primary);
    color: white;
}

.badge-warning {
    background: var(--warning);
    color: white;
}

.fallback-warning {
    background: linear-gradient(135deg, #fef3c7, #fed7aa);
    border: 2px solid var(--warning);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 2rem 0;
    display: flex;
    gap: 1rem;
    align-items: start;
}

.warning-icon {
    font-size: 2rem;
}

.warning-content h2 {
    margin-top: 0;
    color: #92400e;
}

.warning-content p {
    color: #78350f;
    margin: 0.5rem 0;
}

.warning-content code {
    background: #fbbf24;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-family: monospace;
}

section {
    background: var(--bg-primary);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px var(--shadow);
}

.executive-summary {
    font-size: 1.125rem;
    line-height: 1.8;
    color: var(--text-primary);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    background: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: var(--radius);
    text-align: center;
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary);
    display: block;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-top: 0.5rem;
}

.timeline {
    position: relative;
    padding: 2rem 0;
}

.timeline-bar {
    width: 100%;
    height: 4px;
    background: var(--bg-tertiary);
    border-radius: 2px;
    position: relative;
    margin: 2rem 0;
}

.timeline-marker {
    position: absolute;
    width: 12px;
    height: 12px;
    background: var(--primary);
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: pointer;
    transition: all 0.2s;
}

.timeline-marker:hover {
    width: 16px;
    height: 16px;
    background: var(--primary-dark);
}

.chapter-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 2rem;
    transition: box-shadow 0.3s;
}

.chapter-card:hover {
    box-shadow: 0 4px 12px var(--shadow);
}

.chapter-title {
    font-size: 1.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.chapter-date-range {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-bottom: 1rem;
}

.chapter-narrative {
    font-size: 1.125rem;
    line-height: 1.8;
    margin: 1.5rem 0;
    color: var(--text-primary);
}

.chapter-insights {
    background: var(--bg-secondary);
    border-left: 4px solid var(--primary);
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.25rem;
}

.chapter-insights ul {
    list-style: none;
    padding-left: 0;
}

.chapter-insights li {
    padding: 0.5rem 0;
    position: relative;
    padding-left: 1.5rem;
}

.chapter-insights li:before {
    content: "•";
    position: absolute;
    left: 0;
    color: var(--primary);
    font-size: 1.5rem;
    line-height: 1;
}

.thumbnail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.thumbnail {
    aspect-ratio: 1;
    border-radius: 0.5rem;
    overflow: hidden;
    position: relative;
    background: var(--bg-tertiary);
}

.thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.blurred-container {
    position: relative;
    cursor: pointer;
}

.blurred {
    filter: blur(20px);
    transition: filter 0.3s;
}

.blurred-container:hover .blurred {
    filter: blur(15px);
}

.blurred-container.revealed .blurred {
    filter: blur(0);
}

.reveal-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    transition: opacity 0.3s;
}

.blurred-container.revealed .reveal-overlay {
    opacity: 0;
    pointer-events: none;
}

.sensitivity-badge {
    background: var(--warning);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    margin-top: 0.5rem;
}

.sensitive-placeholder {
    aspect-ratio: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    color: var(--text-tertiary);
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
}

.placeholder-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

footer {
    background: var(--bg-primary);
    border-top: 1px solid var(--border);
    padding: 2rem;
    margin-top: 4rem;
    text-align: center;
    color: var(--text-secondary);
}

.footer-content {
    max-width: 1200px;
    margin: 0 auto;
}

.powered-by {
    font-size: 0.875rem;
    margin-top: 1rem;
    color: var(--text-tertiary);
}

.theme-toggle {
    position: fixed;
    top: 1rem;
    right: 1rem;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 9999px;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1.25rem;
    box-shadow: 0 2px 8px var(--shadow);
    transition: all 0.2s;
    z-index: 1000;
}

.theme-toggle:hover {
    transform: scale(1.1);
}

@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    .container,
    .header-content {
        padding: 1rem;
    }
    
    section {
        padding: 1.5rem;
    }
    
    .stats-grid {
        grid-template-columns: 1fr;
    }
    
    .thumbnail-grid {
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    }
}

@media print {
    .theme-toggle {
        display: none;
    }
    
    body {
        background: white;
    }
    
    section {
        box-shadow: none;
        page-break-inside: avoid;
    }
}

/* Visual Enhancements */
.thumbnail-strip {
    display: flex;
    gap: 0.5rem;
    margin: 1rem 0;
    overflow-x: auto;
    padding-bottom: 0.5rem;
    scrollbar-width: thin;
}

.thumbnail-strip::-webkit-scrollbar {
    height: 4px;
}

.thumbnail-strip::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}

.clickable-thumbnail {
    width: 80px;
    height: 80px;
    min-width: 80px;
    border-radius: 0.5rem;
    object-fit: cover;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 2px solid transparent;
}

.clickable-thumbnail:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px var(--shadow);
    border-color: var(--primary);
}

.visual-themes {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px dashed var(--border);
}

.theme-group {
    margin-bottom: 1rem;
}

.theme-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-tertiary);
    margin-bottom: 0.5rem;
    display: block;
}

.theme-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.vibe-badge {
    background: rgba(99, 102, 241, 0.1);
    color: var(--primary);
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.scene-badge {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.motif-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.motif-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

/* Metadata Panel */
.analysis-metadata {
    background: var(--bg-tertiary);
    border-radius: var(--radius);
    padding: 1rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
}

.metadata-item {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.metadata-label {
    color: var(--text-secondary);
}

.metadata-value {
    font-weight: 600;
}

/* Lightbox */
.lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.9);
    z-index: 2000;
    cursor: zoom-out;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}

.lightbox.active {
    display: flex;
}

.lightbox img {
    max-width: 90%;
    max-height: 90%;
    object-fit: contain;
    border-radius: 4px;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}

.lightbox-close {
    position: absolute;
    top: 1.5rem;
    right: 1.5rem;
    color: white;
    font-size: 2rem;
    cursor: pointer;
}

.timeline-vibe-warm { background: #f59e0b; }
.timeline-vibe-cool { background: #6366f1; }
.timeline-vibe-nature { background: #10b981; }
.timeline-vibe-neutral { background: #9ca3af; }
"""


# =============================================================================
# EMBEDDED JAVASCRIPT
# =============================================================================

EMBEDDED_JS = """
// Theme management
function initTheme() {
    const theme = localStorage.getItem('theme') || 'auto';
    applyTheme(theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('theme', next);
}

function applyTheme(theme) {
    if (theme === 'auto') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        theme = prefersDark ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
}

function updateThemeIcon(theme) {
    const toggle = document.querySelector('.theme-toggle');
    if (toggle) {
        toggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}

// Smooth scroll to chapter
function scrollToChapter(chapterId) {
    const element = document.getElementById(chapterId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Reveal blurred image
function revealImage(container) {
    container.classList.add('revealed');
}

// Copy text to clipboard
function copyText(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!');
    });
}

// Simple toast notification
function showToast(message) {
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: var(--primary);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: var(--radius);
        box-shadow: 0 4px 12px var(--shadow);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initLightbox();
});

// Lightbox functionality
function initLightbox() {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `
        <span class="lightbox-close">&times;</span>
        <img src="" alt="Enlarged view">
    `;
    document.body.appendChild(lightbox);

    lightbox.addEventListener('click', () => {
        lightbox.classList.remove('active');
    });

    // Close on escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') lightbox.classList.remove('active');
    });
}

function openLightbox(src) {
    const lightbox = document.querySelector('.lightbox');
    const img = lightbox.querySelector('img');
    img.src = src;
    lightbox.classList.add('active');
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .timeline-marker {
        transition: transform 0.2s, background 0.3s;
    }
    
    .timeline-marker:hover {
        transform: translate(-50%, -50%) scale(1.5);
    }
`;
document.head.appendChild(style);
"""


# =============================================================================
# TEMPLATES
# =============================================================================

BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="{{ theme }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>{{ css }}</style>
</head>
<body>
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">🌙</div>
    
    {{ header }}
    
    <main class="container">
        {{ executive_summary }}
        
        {% if include_timeline %}
        {{ timeline }}
        {% endif %}
        
        {{ chapters }}
        
        {% if include_platform_insights %}
        {{ platform_insights }}
        {% endif %}
        
        {% if include_gaps %}
        {{ gaps }}
        {% endif %}
        
        {% if include_statistics %}
        {{ statistics }}
        {% endif %}
    </main>
    
    {{ footer }}
    
    <script>{{ js }}</script>
</body>
</html>
"""

FALLBACK_WARNING_TEMPLATE = """
<div class="fallback-warning">
    <div class="warning-icon">⚠️</div>
    <div class="warning-content">
        <h2>Statistics-Only Report</h2>
        <p>This report was generated without AI analysis. You're seeing statistics and year-based organization only.</p>
        <p>To unlock the full Life Story experience with meaningful chapters, narratives, and insights, configure your Gemini API key:</p>
        <code>organizer config set-key</code>
    </div>
</div>
"""

SENSITIVE_PLACEHOLDER_TEMPLATE = """
<div class="sensitive-placeholder" data-action="{{ action }}">
    <div class="placeholder-icon">🔒</div>
    <span>Content hidden ({{ reason }})</span>
</div>
"""

BLURRED_IMAGE_TEMPLATE = """
<div class="blurred-container" onclick="revealImage(this)">
    <img src="{{ src }}" class="blurred" alt="{{ alt }}">
    <div class="reveal-overlay">
        <span>Click to reveal</span>
        <span class="sensitivity-badge">{{ badge }}</span>
    </div>
</div>
"""


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ReportConfig:
    """Configuration for HTML report generation."""

    title: str = "Your Life Story"
    subtitle: Optional[str] = None
    theme: Literal["light", "dark", "auto"] = "auto"
    include_timeline: bool = True
    include_statistics: bool = True
    include_platform_insights: bool = True
    include_gaps_section: bool = True
    include_raw_data: bool = False
    embed_thumbnail_data: bool = False
    max_thumbnails_per_chapter: int = 5
    thumbnail_size: tuple[int, int] = (150, 150)
    safety_settings: Optional[Any] = None  # SafetySettings when available
    show_fallback_warning: bool = True
    show_generation_metadata: bool = True
    show_thumbnails: bool = True
    show_visual_summary: bool = True
    show_cost_transparency: bool = True
    custom_css: Optional[str] = None


@dataclass
class ReportSection:
    """A section of the report."""

    id: str
    title: str
    content: str
    order: int
    visible: bool = True
    css_class: Optional[str] = None


# =============================================================================
# HTML REPORT GENERATOR
# =============================================================================


class HTMLReportGenerator:
    """
    Generates beautiful, self-contained HTML reports from LifeStoryReport data.

    This is the primary user-facing output of the application, showcasing
    AI-generated life narratives in an interactive, visually appealing format.
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        """
        Initialize the HTML report generator.

        Args:
            config: Report configuration (uses defaults if None)
        """
        self._config = config or ReportConfig()

        if Environment is None:
            logger.warning("Jinja2 not available, using simple template substitution")
            self._env = None
        else:
            self._env = Environment(autoescape=True)

        self._templates: Dict[str, Any] = {}

    def generate(self, report: Any, output_path: Optional[Path] = None) -> str:
        """
        Generate HTML report from LifeStoryReport.

        Args:
            report: LifeStoryReport instance (using Any for now until module exists)
            output_path: Optional path to write HTML file

        Returns:
            HTML string
        """
        logger.info("Generating HTML report")

        # Build template context
        context = self._build_context(report)

        # Render sections
        sections = self._render_all_sections(report, context)

        # Assemble final HTML
        html_output = self._assemble_html(sections, context)

        # Write to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_output, encoding="utf-8")
            logger.info(f"Report written to {output_path}")

        return html_output

    def generate_to_file(self, report: Any, output_path: Path) -> Path:
        """
        Generate report and write to file.

        Args:
            report: LifeStoryReport instance
            output_path: Path to write HTML file

        Returns:
            Path to generated file
        """
        self.generate(report, output_path=output_path)
        return Path(output_path)

    def _build_context(self, report: Any) -> Dict[str, Any]:
        """Build template context from report data."""
        return {
            "title": self._config.title,
            "subtitle": self._config.subtitle,
            "theme": self._config.theme,
            "css": EMBEDDED_CSS + (self._config.custom_css or ""),
            "js": EMBEDDED_JS,
            "include_timeline": self._config.include_timeline,
            "include_platform_insights": self._config.include_platform_insights,
            "include_gaps": self._config.include_gaps_section,
            "include_statistics": self._config.include_statistics,
            "report": report,
            "config": self._config,
        }

    def _render_all_sections(self, report: Any, context: Dict[str, Any]) -> List[ReportSection]:
        """Render all report sections."""
        sections = []

        sections.append(
            ReportSection(
                id="header",
                title="Header",
                content=self._render_header(report),
                order=0,
            )
        )

        sections.append(
            ReportSection(
                id="executive_summary",
                title="Executive Summary",
                content=self._render_executive_summary(report),
                order=1,
            )
        )

        if self._config.include_timeline:
            sections.append(
                ReportSection(
                    id="timeline",
                    title="Timeline",
                    content=self._render_timeline(report),
                    order=2,
                )
            )

        sections.append(
            ReportSection(
                id="chapters",
                title="Chapters",
                content=self._render_chapters_section(report),
                order=3,
            )
        )

        sections.append(
            ReportSection(
                id="footer",
                title="Footer",
                content=self._render_footer(report),
                order=99,
            )
        )

        return sections

    def _render_header(self, report: Any) -> str:
        """Render the header section."""
        is_fallback = getattr(report, "is_fallback", False)
        total_memories = getattr(report, "total_memories", 0)
        date_range = getattr(report, "date_range", None)
        analysis_metadata = getattr(report, "analysis_metadata", {})

        fallback_warning = ""
        if is_fallback and self._config.show_fallback_warning:
            fallback_warning = FALLBACK_WARNING_TEMPLATE

        date_range_str = ""
        if date_range:
            start = getattr(date_range, "start", None)
            end = getattr(date_range, "end", None)
            if start and end:
                date_range_str = self._format_date_range(start, end)

        # Meta Panel html
        meta_panel = ""
        if self._config.show_cost_transparency and analysis_metadata:
            depth = analysis_metadata.get("depth_mode", "standard")
            vision_model = analysis_metadata.get("vision_model", "gemini-2.0-flash-exp")
            narrative_model = analysis_metadata.get("narrative_model", "gemini-2.0-flash-exp")
            total_images = analysis_metadata.get("total_images_analyzed", 0)
            est_cost = analysis_metadata.get("total_cost_usd", 0.0)
            budget_cap = analysis_metadata.get("budget_cap_applied", False)
            
            meta_panel = f"""
            <div class="analysis-metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Analysis Depth</span>
                    <span class="metadata-value">{depth.title()}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Images Processed</span>
                    <span class="metadata-value">{total_images}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">AI Model</span>
                    <span class="metadata-value">{vision_model}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Estimated Cost</span>
                    <span class="metadata-value">{self._format_usd(est_cost)}</span>
                </div>
                {f'<div class="metadata-item"><span class="metadata-label">Budget Cap</span><span class="metadata-value">Applied</span></div>' if budget_cap else ''}
            </div>
            """

        return f"""
        <header>
            <div class="header-content">
                <h1>{html_lib.escape(self._config.title)}</h1>
                {f'<p class="subtitle">{html_lib.escape(self._config.subtitle)}</p>' if self._config.subtitle else ''}
                <div>
                    {f'<span class="badge">{date_range_str}</span>' if date_range_str else ''}
                    <span class="badge badge-primary">{total_memories:,} memories</span>
                </div>
                {meta_panel}
                {fallback_warning}
            </div>
        </header>
        """

    def _render_executive_summary(self, report: Any) -> str:
        """Render the executive summary section."""
        summary = getattr(report, "executive_summary", "No summary available.")
        is_fallback = getattr(report, "is_fallback", False)

        if is_fallback:
            summary = "This report contains statistical analysis of your media library organized by year. Configure Gemini API to unlock AI-generated narratives and insights."

        return f"""
        <section id="summary">
            <h2>Your Story</h2>
            <div class="executive-summary">
                {html_lib.escape(summary)}
            </div>
        </section>
        """

    def _render_timeline(self, report: Any) -> str:
        """Render interactive timeline."""
        chapters = getattr(report, "chapters", [])

        if not chapters:
            return ""

        markers_html = ""
        for i, chapter in enumerate(chapters):
            position = (i / max(len(chapters) - 1, 1)) * 100
            title = getattr(chapter, "title", f"Chapter {i+1}")
            
            # Use visual vibe for color coding
            visual_stats = getattr(chapter, "visual_stats", None)
            vibes = getattr(visual_stats, "dominant_vibes", []) if visual_stats else []
            vibe_class = self._get_vibe_class(vibes)
            
            markers_html += f"""
                <div class="timeline-marker {vibe_class}" 
                     style="left: {position}%"
                     onclick="scrollToChapter('chapter-{i}')"
                     title="{html_lib.escape(title)}">
                </div>
            """

        return f"""
        <section id="timeline">
            <h2>Timeline</h2>
            <div class="timeline">
                <div class="timeline-bar">
                    {markers_html}
                </div>
            </div>
        </section>
        """

    def _render_chapter(
        self, chapter: Any, index: int, memories: Optional[List[Any]] = None
    ) -> str:
        """Render a single chapter card."""
        title = getattr(chapter, "title", f"Chapter {index + 1}")
        narrative = getattr(chapter, "narrative", "")
        date_range = getattr(chapter, "date_range", None)
        insights = getattr(chapter, "insights", [])
        
        # New visual elements
        thumbnail_paths = getattr(chapter, "thumbnail_paths", []) or []
        
        date_range_str = ""
        if date_range:
            start = getattr(date_range, "start", None)
            end = getattr(date_range, "end", None)
            if start and end:
                date_range_str = self._format_date_range(start, end)

        # Render Thumbnails
        thumbnails_html = ""
        if self._config.show_thumbnails and thumbnail_paths:
            thumbs = []
            max_thumbs = self._config.max_thumbnails_per_chapter
            for path in thumbnail_paths[:max_thumbs]:
                data_uri = self._generate_thumbnail_data_uri(path, self._config.thumbnail_size)
                if data_uri:
                    thumbs.append(f'<img src="{data_uri}" class="clickable-thumbnail" onclick="openLightbox(\'{data_uri}\')" alt="Snapshot">')
            
            if thumbs:
                thumbnails_html = f'<div class="thumbnail-strip">{" ".join(thumbs)}</div>'

        # Render Visual Summary
        visual_summary_html = self._render_visual_summary(chapter)

        insights_html = ""
        if insights:
            insights_items = "\n".join(
                [f"<li>{html_lib.escape(str(i))}</li>" for i in insights[:5]]
            )
            insights_html = f"""
                <div class="chapter-insights">
                    <h3>Contextual Insights</h3>
                    <ul>{insights_items}</ul>
                </div>
            """

        return f"""
        <div class="chapter-card" id="chapter-{index}">
            <h3 class="chapter-title">{html_lib.escape(title)}</h3>
            {f'<p class="chapter-date-range">{date_range_str}</p>' if date_range_str else ''}
            
            {thumbnails_html}
            
            <div class="chapter-narrative">
                {html_lib.escape(narrative)}
            </div>
            
            {visual_summary_html}
            {insights_html}
        </div>
        """

    def _render_chapters_section(self, report: Any) -> str:
        """Render all chapters."""
        chapters = getattr(report, "chapters", [])

        if not chapters:
            return "<section><h2>Chapters</h2><p>No chapters generated.</p></section>"

        chapters_html = "\n".join(
            [self._render_chapter(chapter, i) for i, chapter in enumerate(chapters)]
        )

        return f"""
        <section id="chapters">
            <h2>Your Life in Chapters</h2>
            {chapters_html}
        </section>
        """

    def _render_footer(self, report: Any) -> str:
        """Render footer section."""
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_used = getattr(report, "model_version", "Unknown")

        metadata_html = ""
        if self._config.show_generation_metadata:
            metadata_html = f"""
                <p>Generated on {generation_time}</p>
                <p class="powered-by">Powered by Google Gemini ({model_used})</p>
            """

        return f"""
        <footer>
            <div class="footer-content">
                <p>Your life story, reconstructed from {getattr(report, 'total_memories', 0):,} memories.</p>
                {metadata_html}
                <p style="font-size: 0.75rem; margin-top: 1rem;">All personal data remains private and local.</p>
            </div>
        </footer>
        """

    def _assemble_html(self, sections: List[ReportSection], context: Dict[str, Any]) -> str:
        """Assemble final HTML from sections."""
        # Sort sections by order
        sections = sorted([s for s in sections if s.visible], key=lambda x: x.order)

        # Build section content map
        section_map = {s.id: s.content for s in sections}

        # Simple template substitution (will use Jinja2 when available)
        template_vars = {
            **context,
            **section_map,
            "executive_summary": section_map.get("executive_summary", ""),
            "timeline": section_map.get("timeline", ""),
            "chapters": section_map.get("chapters", ""),
            "platform_insights": "",
            "gaps": "",
            "statistics": "",
        }

        html_output = BASE_TEMPLATE
        for key, value in template_vars.items():
            html_output = html_output.replace("{{ " + key + " }}", str(value))

        # Handle conditionals (simple approach)
        if not context.get("include_timeline"):
            html_output = html_output.replace("{% if include_timeline %}", "<!--")
            html_output = html_output.replace("{% endif %}", "-->")

        return html_output

    # Utility methods

    def _format_date(self, dt: Any) -> str:
        """Format date for display."""
        if isinstance(dt, datetime):
            return dt.strftime("%B %d, %Y")
        elif isinstance(dt, date):
            return dt.strftime("%B %d, %Y")
        return str(dt)

    def _format_date_range(self, start: Any, end: Any) -> str:
        """Format date range for display."""
        if isinstance(start, (date, datetime)) and isinstance(end, (date, datetime)):
            start_str = start.strftime("%B %Y")
            end_str = end.strftime("%B %Y")
            if start_str == end_str:
                return start_str
            return f"{start_str} - {end_str}"
        return f"{start} - {end}"

    def _escape_html(self, text: str) -> str:
        """Safely escape HTML."""
        return html_lib.escape(text)

    def _generate_thumbnail_data_uri(self, path: str, size: tuple[int, int]) -> Optional[str]:
        """Load image, resize, and convert to base64 data URI."""
        if Image is None:
            return None
        
        try:
            img_path = Path(path)
            if not img_path.exists():
                return None
            
            with Image.open(img_path) as img:
                img.thumbnail(size)
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                    
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=80)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.warning(f"Error generating thumbnail for {path}: {e}")
            return None

    def _render_visual_summary(self, chapter: Any) -> str:
        """Render the visual themes section for a chapter."""
        if not self._config.show_visual_summary:
            return ""
            
        visual_stats = getattr(chapter, "visual_stats", None)
        if not visual_stats:
            return ""
            
        scenes = getattr(visual_stats, "dominant_scenes", [])
        vibes = getattr(visual_stats, "dominant_vibes", [])
        motifs = getattr(visual_stats, "recurring_motifs", [])
        
        if not (scenes or vibes or motifs):
            return ""
            
        html = '<div class="visual-themes">'
        
        # Scenes
        if scenes:
            html += '<div class="theme-group"><span class="theme-label">Atmosphere & Setting</span><div class="theme-badges">'
            for scene, count in scenes[:3]:
                emoji = SCENE_EMOJI.get(scene.lower(), "📍")
                html += f'<span class="badge scene-badge">{emoji} {scene.lower()}</span>'
            html += '</div></div>'
            
        # Vibes
        if vibes:
            html += '<div class="theme-group"><span class="theme-label">Emotional Tone</span><div class="theme-badges">'
            for vibe, count in vibes[:3]:
                emoji = VIBE_EMOJI.get(vibe.lower(), "✨")
                html += f'<span class="badge vibe-badge">{emoji} {vibe.lower()}</span>'
            html += '</div></div>'
            
        # Motifs
        if motifs:
            html += '<div class="theme-group"><span class="theme-label">Visual Motifs</span><div class="motif-list">'
            for motif, count in motifs[:6]:
                # motif can be "coffee" or "coffee:3" or a tuple
                motif_name = motif[0] if isinstance(motif, (list, tuple)) else str(motif).split(":")[0]
                emoji = MOTIF_EMOJI.get(motif_name.lower(), "•")
                html += f'<span class="motif-item">{emoji} {motif_name.lower()}</span>'
            html += '</div></div>'
            
        html += '</div>'
        return html

    def _format_usd(self, amount: float) -> str:
        """Format USD amount."""
        return f"${amount:,.2f}"

    def _get_vibe_class(self, vibes: List[Any]) -> str:
        """Determine CSS class for timeline marker based on vibe."""
        if not vibes:
            return "timeline-vibe-neutral"
            
        # Extract name if it's a tuple (vibe, count)
        top_vibe = vibes[0][0] if isinstance(vibes[0], (list, tuple)) else str(vibes[0])
        top_vibe = top_vibe.lower()
        
        warm_vibes = ["social", "festive", "energetic", "intimate", "adventure"]
        cool_vibes = ["professional", "reflective", "calm", "official"]
        nature_vibes = ["nature", "beach", "outdoor"]
        
        if top_vibe in warm_vibes: return "timeline-vibe-warm"
        if top_vibe in cool_vibes: return "timeline-vibe-cool"
        if top_vibe in nature_vibes: return "timeline-vibe-nature"
        
        return "timeline-vibe-neutral"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_report(
    report: Any,
    output_path: Path,
    config: Optional[ReportConfig] = None,
) -> Path:
    """
    Generate HTML report and write to file.

    Args:
        report: LifeStoryReport instance
        output_path: Path to write HTML file
        config: Optional report configuration

    Returns:
        Path to generated file
    """
    generator = HTMLReportGenerator(config=config)
    return generator.generate_to_file(report, output_path)


def generate_report_string(
    report: Any,
    config: Optional[ReportConfig] = None,
) -> str:
    """
    Generate HTML report as string without writing to file.

    Args:
        report: LifeStoryReport instance
        config: Optional report configuration

    Returns:
        HTML string
    """
    generator = HTMLReportGenerator(config=config)
    return generator.generate(report)
