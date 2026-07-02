"""
Alice Scripting System — collapsed to a tiny prompt builder.

The old StateStorage/ScriptEngine/StateTranslator stack has been archived to
master_archive/simplify_2026_05_05/scripting/. ScriptIntegration is now a
thin shim that loads `templates/base_chat.txt` and substitutes the few
narrative slots Alice's prompt actually consumes.
"""

from .script_integration import ScriptIntegration

__all__ = ["ScriptIntegration"]
