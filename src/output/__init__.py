"""Output generation module for Life Story reports."""

from src.output.html_report import (
    HTMLReportGenerator,
    ReportConfig,
    ReportSection,
    generate_report,
    generate_report_string,
)

__all__ = [
    "HTMLReportGenerator",
    "ReportConfig",
    "ReportSection",
    "generate_report",
    "generate_report_string",
]
