"""
Alice - AI Streamer Personality System
=============================================

A sassy, brilliant AI built on a fine-tuned language model with comprehensive
safety, memory, and entertainment capabilities.

Core Components:
- core: Personality model, inference, memory systems
- subroutines: Safety layers and behavior control
- interfaces: User interaction systems
- streaming: Streaming and avatar components
- tests: Comprehensive testing suite

Author: Rin
Version: 1.1.0-v3 (Personality Override)
"""

# Fix Windows console encoding for emoji support - MUST BE FIRST
import sys
import os

# Force UTF-8 for Windows - must happen before ANY output
if sys.platform == 'win32':
    # Set environment for subprocess compatibility
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    # Only wrap if we have a real console (not already wrapped/redirected)
    try:
        if hasattr(sys.stdout, 'buffer') and not hasattr(sys.stdout, '_wrapped_for_utf8'):
            import io
            # Use reconfigure if available (Python 3.7+)
            if hasattr(sys.stdout, 'reconfigure'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    # Fallback to wrapping
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            else:
                # Python 3.6 and earlier
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

            sys.stdout._wrapped_for_utf8 = True
            sys.stderr._wrapped_for_utf8 = True
    except Exception as e:
        # If wrapping fails, suppress emojis by replacing them
        pass

__version__ = "1.1.0-v3"
__author__ = "Rin"

# Core imports
# ARCHIVED (v4): Personality, inference, and harm-classification (WinnieThePooh) all retired.
# Alice's content filtering lives in alice.core.fairy now (TOS-compliance only).

__all__ = []