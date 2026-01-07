"""Report generator for Digital Life Narrative AI.

Generates beautiful, interactive HTML reports and JSON exports from
LifeStoryReport data. The HTML report is self-contained with inline
CSS and JavaScript, requiring no external dependencies.

The report brings the AI-generated life story to life with:
- Interactive chapter timeline
- Beautifully styled narratives
- Platform insights visualization
- Clear fallback mode indication
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

from organizer.models import LifeStoryReport

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ReportFormat(Enum):
    """Output format for the report."""

    HTML = "html"
    JSON = "json"
    BOTH = "both"


# =============================================================================
# Helper Functions
# =============================================================================


def format_date_range(start: date | None, end: date | None) -> str:
    """Format a date range for display.

    Args:
        start: Start date.
        end: End date.

    Returns:
        Formatted string like "Jan 2019 - Dec 2023".
    """
    if not start or not end:
        return "Unknown period"

    start_str = start.strftime("%b %Y")
    end_str = end.strftime("%b %Y")

    if start_str == end_str:
        return start_str

    return f"{start_str} – {end_str}"


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Pluralize a word based on count.

    Args:
        count: The count to check.
        singular: Singular form.
        plural: Plural form (defaults to singular + 's').

    Returns:
        Appropriate form with count.
    """
    if plural is None:
        plural = singular + "s"

    if count == 1:
        return f"{count} {singular}"
    return f"{count:,} {plural}"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis.

    Args:
        text: Text to truncate.
        max_length: Maximum length.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rsplit(" ", 1)[0] + "..."


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        /* CSS Variables for theming */
        :root {
            --bg-primary: #fafafa;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f5f5f5;
            --text-primary: #1a1a1a;
            --text-secondary: #666666;
            --text-muted: #999999;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --accent-bg: #eef2ff;
            --border: #e5e5e5;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.1);
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --warning-bg: #fef3c7;
            --warning-border: #f59e0b;
            --warning-text: #92400e;
            --success: #10b981;
            --chapter-colors: #6366f1, #8b5cf6, #a855f7, #d946ef, #ec4899, #f43f5e;
        }

        [data-theme="dark"] {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #262626;
            --text-primary: #fafafa;
            --text-secondary: #a3a3a3;
            --text-muted: #737373;
            --border: #404040;
            --shadow: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
            --accent-bg: #1e1b4b;
            --warning-bg: #451a03;
            --warning-border: #d97706;
            --warning-text: #fcd34d;
        }

        /* Reset & Base */
        *, *::before, *::after { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            margin: 0;
            padding: 0;
        }

        /* Typography */
        h1, h2, h3, h4 { 
            font-weight: 600; 
            line-height: 1.3; 
            margin: 0 0 1rem 0;
        }
        
        h1 { font-size: 2.5rem; }
        h2 { font-size: 1.75rem; }
        h3 { font-size: 1.25rem; }
        p { margin: 0 0 1rem 0; }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* Layout */
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 0 1.5rem;
        }

        /* Header */
        .header {
            background: var(--gradient);
            color: white;
            padding: 4rem 0 3rem;
            text-align: center;
            position: relative;
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        .header .subtitle {
            font-size: 1.25rem;
            opacity: 0.9;
        }

        .header .meta {
            margin-top: 1.5rem;
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .theme-toggle {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            font-size: 1.2rem;
            transition: transform 0.3s;
        }
        
        .theme-toggle:hover { transform: scale(1.1); }

        /* Fallback Warning */
        .fallback-warning {
            background: var(--warning-bg);
            border: 2px solid var(--warning-border);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 2rem 0;
            color: var(--warning-text);
        }

        .fallback-warning h3 {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .fallback-warning code {
            background: rgba(0,0,0,0.1);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }

        .stat-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: var(--shadow);
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }

        .stat-card .label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        /* Executive Summary */
        .executive-summary {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: var(--shadow-lg);
        }

        .executive-summary h2 {
            color: var(--accent);
            margin-bottom: 1.5rem;
        }

        .executive-summary .narrative {
            font-size: 1.1rem;
            color: var(--text-primary);
        }

        .executive-summary .narrative p {
            margin-bottom: 1.25rem;
        }

        /* Timeline */
        .timeline-section {
            margin: 3rem 0;
        }

        .timeline {
            display: flex;
            overflow-x: auto;
            gap: 0;
            padding: 2rem 0;
            position: relative;
        }

        .timeline::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--border);
            z-index: 0;
        }

        .timeline-item {
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0 1rem;
            cursor: pointer;
            position: relative;
            z-index: 1;
            transition: transform 0.2s;
        }

        .timeline-item:hover { transform: translateY(-5px); }

        .timeline-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent);
            border: 4px solid var(--bg-primary);
            box-shadow: var(--shadow);
        }

        .timeline-label {
            margin-top: 0.75rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-align: center;
            max-width: 100px;
        }

        .timeline-title {
            font-weight: 600;
            font-size: 0.8rem;
            color: var(--text-primary);
        }

        /* Chapters */
        .chapters-section {
            margin: 3rem 0;
        }

        .chapter-card {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--accent);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .chapter-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .chapter-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .chapter-title {
            margin: 0;
            font-size: 1.5rem;
        }

        .chapter-dates {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .chapter-meta {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .tag {
            background: var(--accent-bg);
            color: var(--accent);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .tag.count {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }

        .chapter-narrative {
            margin: 1.5rem 0;
            color: var(--text-primary);
            font-size: 1.05rem;
        }

        .chapter-events {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }

        .chapter-events h4 {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        .events-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .events-list li {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .events-list li::before {
            content: '✦';
            color: var(--accent);
        }

        /* Platform Insights */
        .insights-section {
            margin: 3rem 0;
        }

        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .insight-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }

        .insight-card h3 {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .platform-icon {
            width: 24px;
            height: 24px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
        }

        .platform-snapchat { background: #FFFC00; color: black; }
        .platform-google_photos { background: #4285f4; color: white; }
        .platform-facebook { background: #1877f2; color: white; }
        .platform-instagram { background: linear-gradient(45deg, #f09433, #e6683c, #dc2743, #cc2366, #bc1888); color: white; }
        .platform-local { background: #6b7280; color: white; }

        .insight-pattern {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }

        .insight-meta {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Patterns & Gaps */
        .analysis-section {
            margin: 3rem 0;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
        }

        .analysis-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }

        .analysis-card h3 {
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .pattern-list, .gap-list, .notes-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .pattern-list li, .notes-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .pattern-list li:last-child, .notes-list li:last-child {
            border-bottom: none;
        }

        .gap-item {
            padding: 0.75rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 0.75rem;
        }

        .gap-dates {
            font-weight: 600;
            color: var(--text-primary);
        }

        .gap-duration {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Footer */
        .footer {
            background: var(--bg-tertiary);
            padding: 3rem 0;
            margin-top: 4rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .footer .privacy-note {
            max-width: 600px;
            margin: 1rem auto 0;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .animate-in {
            animation: fadeIn 0.6s ease-out forwards;
        }

        /* Responsive */
        @media (max-width: 640px) {
            .header h1 { font-size: 2rem; }
            .executive-summary { padding: 1.5rem; }
            .chapter-card { padding: 1.5rem; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }

        /* Section Headers */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .section-header h2 {
            margin: 0;
        }

        .section-icon {
            font-size: 1.5rem;
        }

        /* Print styles */
        @media print {
            .theme-toggle, .timeline { display: none; }
            .chapter-card { break-inside: avoid; }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark mode">🌙</button>
        <div class="container">
            <h1>{{ title }}</h1>
            <p class="subtitle">{{ subtitle }}</p>
            <p class="meta">
                {% if report.date_range %}
                {{ format_date_range(report.date_range[0], report.date_range[1]) }}
                {% endif %}
                · {{ pluralize(report.total_media_analyzed, 'memory', 'memories') }}
            </p>
        </div>
    </header>

    <main class="container">
        {% if report.is_fallback_mode %}
        <!-- Fallback Warning -->
        <div class="fallback-warning animate-in">
            <h3>⚠️ AI Analysis Unavailable</h3>
            <p>This report was generated in <strong>fallback mode</strong> without AI-powered narrative analysis. You're seeing statistics only.</p>
            <p>To unlock the full life story experience with rich narratives and insights:</p>
            <p><code>organizer configure --set-key</code></p>
        </div>
        {% endif %}

        <!-- Stats Overview -->
        <div class="stats-grid animate-in">
            <div class="stat-card">
                <div class="value">{{ report.total_media_analyzed | format_number }}</div>
                <div class="label">Total Memories</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ report.chapters | length }}</div>
                <div class="label">Life Chapters</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ report.years_covered or 0 }}</div>
                <div class="label">Years Covered</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ report.platform_insights | length or platforms_count }}</div>
                <div class="label">Platforms</div>
            </div>
        </div>

        <!-- Executive Summary -->
        <section class="executive-summary animate-in">
            <h2>📖 Your Story</h2>
            <div class="narrative">
                {{ report.executive_summary | safe | markdown }}
            </div>
        </section>

        {% if report.chapters %}
        <!-- Timeline -->
        <section class="timeline-section">
            <div class="section-header">
                <span class="section-icon">📅</span>
                <h2>Life Timeline</h2>
            </div>
            <div class="timeline">
                {% for chapter in report.chapters %}
                <div class="timeline-item" onclick="scrollToChapter('chapter-{{ loop.index }}')" style="--color: {{ chapter_colors[loop.index0 % chapter_colors | length] }}">
                    <div class="timeline-dot" style="background: var(--color)"></div>
                    <div class="timeline-label">
                        <div class="timeline-title">{{ chapter.title | truncate(20) }}</div>
                        <div>{{ chapter.start_date.year }}</div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </section>

        <!-- Chapters -->
        <section class="chapters-section">
            <div class="section-header">
                <span class="section-icon">📚</span>
                <h2>Life Chapters</h2>
            </div>
            {% for chapter in report.chapters %}
            <article class="chapter-card animate-in" id="chapter-{{ loop.index }}" style="border-left-color: {{ chapter_colors[loop.index0 % chapter_colors | length] }}">
                <div class="chapter-header">
                    <div>
                        <h3 class="chapter-title">{{ chapter.title }}</h3>
                        <div class="chapter-dates">{{ format_date_range(chapter.start_date, chapter.end_date) }}</div>
                    </div>
                    <div class="chapter-meta">
                        {% for theme in chapter.themes[:3] %}
                        <span class="tag">{{ theme }}</span>
                        {% endfor %}
                        <span class="tag count">{{ pluralize(chapter.media_count, 'item') }}</span>
                    </div>
                </div>
                <div class="chapter-narrative">
                    {{ chapter.narrative | safe | markdown }}
                </div>
                {% if chapter.key_events %}
                <div class="chapter-events">
                    <h4>Key Moments</h4>
                    <ul class="events-list">
                        {% for event in chapter.key_events[:5] %}
                        <li>{{ event }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </article>
            {% endfor %}
        </section>
        {% endif %}

        {% if report.platform_insights %}
        <!-- Platform Insights -->
        <section class="insights-section">
            <div class="section-header">
                <span class="section-icon">📱</span>
                <h2>Platform Insights</h2>
            </div>
            <div class="insights-grid">
                {% for insight in report.platform_insights %}
                <div class="insight-card">
                    <h3>
                        <span class="platform-icon platform-{{ insight.platform.value }}">
                            {{ platform_icons.get(insight.platform.value, '📷') }}
                        </span>
                        {{ insight.platform.value | title | replace('_', ' ') }}
                    </h3>
                    <p class="insight-pattern">{{ insight.usage_pattern }}</p>
                    <div class="insight-meta">
                        {% if insight.peak_years %}
                        Peak years: {{ insight.peak_years | join(', ') }}
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </section>
        {% endif %}

        <!-- Patterns & Gaps -->
        <section class="analysis-section">
            <div class="section-header">
                <span class="section-icon">🔍</span>
                <h2>Analysis Details</h2>
            </div>
            <div class="analysis-grid">
                {% if report.detected_patterns %}
                <div class="analysis-card">
                    <h3>📊 Detected Patterns</h3>
                    <ul class="pattern-list">
                        {% for pattern in report.detected_patterns %}
                        <li>{{ pattern }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if report.data_gaps %}
                <div class="analysis-card">
                    <h3>📭 Data Gaps</h3>
                    <div class="gap-list">
                        {% for gap in report.data_gaps[:5] %}
                        <div class="gap-item">
                            <div class="gap-dates">{{ format_date_range(gap.start_date, gap.end_date) }}</div>
                            <div class="gap-duration">{{ gap.gap_days }} days</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}

                {% if report.data_quality_notes %}
                <div class="analysis-card">
                    <h3>📋 Data Quality</h3>
                    <ul class="notes-list">
                        {% for note in report.data_quality_notes %}
                        <li>{{ note }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>
                Generated on {{ report.generated_at.strftime('%B %d, %Y at %I:%M %p') }}
                {% if report.ai_model_used %}
                · AI Model: {{ report.ai_model_used }}
                {% endif %}
            </p>
            <p class="privacy-note">
                🔒 This report was generated locally. Your media and personal data never left your device
                {% if not report.is_fallback_mode %}(only anonymized metadata was sent to the AI for narrative generation){% endif %}.
            </p>
        </div>
    </footer>

    <script>
        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            document.querySelector('.theme-toggle').textContent = newTheme === 'dark' ? '☀️' : '🌙';
        }

        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
            document.querySelector('.theme-toggle').textContent = savedTheme === 'dark' ? '☀️' : '🌙';
        }

        // Smooth scroll to chapter
        function scrollToChapter(id) {
            const element = document.getElementById(id);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.style.boxShadow = '0 0 0 3px var(--accent)';
                setTimeout(() => {
                    element.style.boxShadow = '';
                }, 2000);
            }
        }

        // Animate elements on scroll
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.chapter-card, .insight-card, .analysis-card').forEach(el => {
            observer.observe(el);
        });
    </script>
</body>
</html>"""


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """Generates HTML and JSON reports from LifeStoryReport data.

    Creates beautiful, self-contained HTML reports with inline CSS/JS,
    or JSON exports for programmatic use.

    Attributes:
        output_format: The format(s) to generate.

    Example:
        ```python
        generator = ReportGenerator(ReportFormat.HTML)
        paths = generator.generate(report, Path("output/my_story"))
        print(f"Generated: {paths}")
        ```
    """

    CHAPTER_COLORS = [
        "#6366f1",
        "#8b5cf6",
        "#a855f7",
        "#d946ef",
        "#ec4899",
        "#f43f5e",
        "#ef4444",
        "#f97316",
        "#eab308",
        "#22c55e",
        "#14b8a6",
        "#0ea5e9",
    ]

    PLATFORM_ICONS = {
        "snapchat": "👻",
        "google_photos": "📸",
        "facebook": "📘",
        "instagram": "📷",
        "onedrive": "☁️",
        "local": "💾",
        "unknown": "❓",
    }

    def __init__(self, output_format: ReportFormat = ReportFormat.HTML) -> None:
        """Initialize the report generator.

        Args:
            output_format: Format(s) to generate.
        """
        self.output_format = output_format
        self._env = self._create_jinja_env()
        logger.debug(f"ReportGenerator initialized with format: {output_format.value}")

    def _create_jinja_env(self) -> Environment:
        """Create Jinja2 environment with custom filters.

        Returns:
            Configured Jinja2 Environment.
        """
        env = Environment(loader=BaseLoader())

        # Custom filters
        env.filters["format_number"] = lambda x: f"{x:,}"
        env.filters["truncate"] = truncate_text
        env.filters["markdown"] = self._simple_markdown

        # Global functions
        env.globals["format_date_range"] = format_date_range
        env.globals["pluralize"] = pluralize

        return env

    def _simple_markdown(self, text: str) -> str:
        """Simple markdown-to-HTML conversion.

        Handles basic formatting without external dependencies.

        Args:
            text: Markdown text.

        Returns:
            HTML string.
        """
        if not text:
            return ""

        # Convert line breaks to paragraphs
        paragraphs = text.split("\n\n")
        html_parts = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Handle bold
            para = para.replace("**", "<strong>", 1)
            while "**" in para:
                para = para.replace("**", "</strong>", 1)
                para = para.replace("**", "<strong>", 1)
            para = para.replace("**", "</strong>")

            # Handle italic
            para = para.replace("*", "<em>", 1)
            while "*" in para:
                para = para.replace("*", "</em>", 1)
                para = para.replace("*", "<em>", 1)
            para = para.replace("*", "</em>")

            # Handle code
            para = para.replace("`", "<code>", 1)
            while "`" in para:
                para = para.replace("`", "</code>", 1)
                para = para.replace("`", "<code>", 1)
            para = para.replace("`", "</code>")

            # Wrap in paragraph
            html_parts.append(f"<p>{para}</p>")

        return "\n".join(html_parts)

    def generate(
        self,
        report: LifeStoryReport,
        output_path: Path,
    ) -> list[Path]:
        """Generate report file(s).

        Args:
            report: The LifeStoryReport to render.
            output_path: Base path for output (without extension).

        Returns:
            List of created file paths.
        """
        created_files = []

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_format in (ReportFormat.HTML, ReportFormat.BOTH):
            html_path = self._generate_html(report, output_path)
            created_files.append(html_path)
            logger.info(f"Generated HTML report: {html_path}")

        if self.output_format in (ReportFormat.JSON, ReportFormat.BOTH):
            json_path = self._generate_json(report, output_path)
            created_files.append(json_path)
            logger.info(f"Generated JSON export: {json_path}")

        return created_files

    def _generate_html(
        self,
        report: LifeStoryReport,
        output_path: Path,
    ) -> Path:
        """Generate the HTML report.

        Args:
            report: The report data.
            output_path: Base output path.

        Returns:
            Path to created HTML file.
        """
        html_path = output_path.with_suffix(".html")

        # Prepare template context
        context = self._prepare_context(report)

        # Render template
        template = self._env.from_string(HTML_TEMPLATE)
        html_content = template.render(**context)

        # Write file
        html_path.write_text(html_content, encoding="utf-8")

        return html_path

    def _generate_json(
        self,
        report: LifeStoryReport,
        output_path: Path,
    ) -> Path:
        """Generate JSON export.

        Args:
            report: The report data.
            output_path: Base output path.

        Returns:
            Path to created JSON file.
        """
        json_path = output_path.with_suffix(".json")

        # Use Pydantic's built-in JSON serialization
        json_content = report.model_dump_json(indent=2)

        # Write file
        json_path.write_text(json_content, encoding="utf-8")

        return json_path

    def _prepare_context(self, report: LifeStoryReport) -> dict[str, Any]:
        """Prepare template context from report.

        Args:
            report: The life story report.

        Returns:
            Context dictionary for template.
        """
        # Determine title
        if report.is_fallback_mode:
            title = "Your Life Story (Preview)"
            subtitle = "Statistical overview of your media collection"
        else:
            title = "Your Life Story"
            subtitle = "An AI-powered narrative of your journey"

        # Count unique platforms from chapters or calculate
        platforms_count = (
            len(set(insight.platform.value for insight in report.platform_insights))
            if report.platform_insights
            else 1
        )

        return {
            "report": report,
            "title": title,
            "subtitle": subtitle,
            "platforms_count": platforms_count,
            "chapter_colors": self.CHAPTER_COLORS,
            "platform_icons": self.PLATFORM_ICONS,
            "format_date_range": format_date_range,
            "pluralize": pluralize,
        }


# =============================================================================
# Module-Level Functions
# =============================================================================


def generate_report(
    report: LifeStoryReport,
    output_path: Path,
    output_format: ReportFormat = ReportFormat.HTML,
) -> list[Path]:
    """Convenience function to generate a report.

    Args:
        report: The LifeStoryReport to render.
        output_path: Base path for output.
        output_format: Format(s) to generate.

    Returns:
        List of created file paths.
    """
    generator = ReportGenerator(output_format)
    return generator.generate(report, output_path)
