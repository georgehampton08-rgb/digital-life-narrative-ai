"""Digital Life Narrative AI - Source Package.

DEPRECATION WARNING:
The 'src.' prefix is no longer used for imports. 
The package has been renamed to 'dlnai'.

If you are seeing this error, it means you are attempting to import from the old namespace.
Instead of: `from src.core import ...`
Use:        `from dlnai.core import ...`
"""

raise ImportError(
    "The 'src' prefix is deprecated and no longer supported as a package root. "
    "Please use the 'dlnai' package instead (e.g., 'import dlnai.core')."
)
