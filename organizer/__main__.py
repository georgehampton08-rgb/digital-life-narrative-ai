"""Entry point for running the package as a module.

Allows running the package via:
    python -m organizer

This is equivalent to running the CLI directly.
"""

import sys


def main() -> int:
    """Main entry point with error handling.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    try:
        from organizer.cli import cli

        # Click handles sys.exit internally
        cli()
        return 0

    except KeyboardInterrupt:
        # User interrupted with Ctrl+C
        print("\n\nInterrupted by user.")
        return 130

    except ImportError as e:
        # Missing dependencies
        print(f"\n❌ Missing dependency: {e}")
        print("\nPlease install dependencies with:")
        print("  poetry install")
        print("  # or")
        print("  pip install -e .")
        return 1

    except Exception as e:
        # Unexpected errors
        print(f"\n❌ Unexpected error: {e}")
        print("\nThis might be a bug. Please report it at:")
        print("  https://github.com/georgehampton08-rgb/digital-life-narrative-ai/issues")
        print("\nFor debugging, run with --verbose flag:")
        print("  organizer --verbose <command>")
        return 1


if __name__ == "__main__":
    sys.exit(main())
