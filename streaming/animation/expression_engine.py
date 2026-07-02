#!/usr/bin/env python3
"""
Expression Engine - Triggers VTS hotkey expressions based on emotions.

Maps emotions (from EmotionBERT or Alice's state) to VTS expression hotkeys.
Expressions layer on top of motion clips and auto-disable when emotion returns to neutral.

On init, queries VTS for actual hotkey IDs (UUIDs) by matching display names.
"""

import asyncio
import time
import random
from typing import Optional, Dict, List
from dataclasses import dataclass


# Our internal expression names → expected VTS hotkey display names
# These get matched against what VTS reports via HotkeysInCurrentModelRequest
EXPRESSION_DISPLAY_NAMES = {
    # Poses/Gestures
    'praying': 'praying',
    'covering_chest': 'covering_chest',
    'heart_gesture': 'heart_gesture',
    'holding_star': 'holding_star',

    # Eye Expressions
    'white_eyes': 'white_eyes',
    'black_eyes': 'black_eyes',
    'heart_eyes': 'heart_eyes',
    'star_eyes': 'star_eyes',
    'dizzy': 'dizzy',

    # Face Expressions
    'crying': 'crying',
    'blushing': 'blushing',
    'dark_face': 'dark_face',
    'zzz_sleepy': 'ZZZ',
    'speechless': 'speechless',
    'angry': 'angry',
    'tongue_out': 'tongue_out',
}


# Emotion → Expression mapping (based on user corrections)
EMOTION_EXPRESSIONS = {
    'flirty': ['heart_eyes', 'heart_gesture', 'tongue_out'],
    'excited': ['star_eyes'],
    'sad': ['crying'],
    'tired': ['zzz_sleepy'],
    'angry': ['angry'],  # Also for pouting
    'disappointed': ['dark_face'],
    'surprised': ['speechless', 'dizzy'],
    'embarrassed': ['blushing'],
    'happy': [],  # No specific expression - just motion/smiling
    'neutral': [],  # No expression
    'thinking': [],  # No expression
    'confused': ['speechless'],
}


@dataclass
class ActiveExpression:
    """Currently active expression."""
    name: str
    hotkey_id: str
    triggered_at: float
    emotion: str


class ExpressionEngine:
    """
    Manages VTS expression hotkeys based on emotion state.

    On init, queries VTS for real hotkey IDs by matching display names.
    This is required because VTS API needs UUIDs, not keyboard shortcut strings.

    Usage:
        engine = ExpressionEngine(vts_connection)
        await engine.init_hotkeys()  # Must call after VTS connected
        await engine.update_emotion('happy')
        await engine.update_emotion('neutral')  # Clears expressions
    """

    def __init__(self, vts_connection):
        self.vts = vts_connection
        self.current_emotion: str = 'neutral'
        self.active_expressions: List[ActiveExpression] = []
        self.last_emotion_change: float = time.time()
        # Maps our internal name → actual VTS hotkey UUID
        self.hotkey_ids: Dict[str, str] = {}
        self.initialized = False

    async def init_hotkeys(self):
        """
        Query VTS for all hotkeys and match them to our expression names.
        Must be called after VTS connection is established.
        """
        hotkeys = await self.vts.list_hotkeys()
        if not hotkeys:
            print("[Expression Engine] WARNING: No hotkeys found in VTS model")
            return

        print(f"\n[Expression Engine] Found {len(hotkeys)} hotkeys in model:")
        for hk in hotkeys:
            print(f"    '{hk.get('name', '')}' (type: {hk.get('type', '?')})")

        # Hotkeys that must NEVER be triggered (break the model)
        BLOCKED_HOTKEYS = {'flying_head', 'flying head', 'shrink'}

        # Build a normalized name → hotkeyID lookup from VTS
        vts_lookup = {}
        for hk in hotkeys:
            name = hk.get("name", "")
            hk_id = hk.get("hotkeyID", "")
            hk_type = hk.get("type", "")
            if name and hk_id:
                # Normalize: lowercase, strip whitespace
                normalized_name = name.lower().strip()
                if normalized_name in BLOCKED_HOTKEYS:
                    continue
                vts_lookup[normalized_name] = (hk_id, name, hk_type)

        # Match our expressions to VTS hotkeys
        matched = 0
        for our_name, display_name in EXPRESSION_DISPLAY_NAMES.items():
            normalized = display_name.lower().strip()
            if normalized in vts_lookup:
                hk_id, original_name, hk_type = vts_lookup[normalized]
                self.hotkey_ids[our_name] = hk_id
                matched += 1
            else:
                # Try fuzzy: check if display name is contained in any VTS name
                found = False
                for vts_name, (hk_id, original_name, hk_type) in vts_lookup.items():
                    if normalized in vts_name or vts_name in normalized:
                        self.hotkey_ids[our_name] = hk_id
                        matched += 1
                        found = True
                        print(f"  Fuzzy match: {our_name} → '{original_name}'")
                        break
                if not found:
                    print(f"  No match for: {our_name} (expected '{display_name}')")

        print(f"  Matched {matched}/{len(EXPRESSION_DISPLAY_NAMES)} expressions")

        # Show which emotion expressions are available
        for emotion, expr_list in EMOTION_EXPRESSIONS.items():
            if expr_list:
                available = [e for e in expr_list if e in self.hotkey_ids]
                if available:
                    print(f"  {emotion}: {', '.join(available)}")
                else:
                    missing = [e for e in expr_list if e not in self.hotkey_ids]
                    print(f"  {emotion}: MISSING - {', '.join(missing)}")

        self.initialized = True

    async def update_emotion(self, emotion: str):
        """Update current emotion and trigger/clear expressions accordingly."""
        emotion = emotion.lower()

        if emotion == self.current_emotion:
            return

        print(f"\n[Expression Engine] Emotion: {self.current_emotion} → {emotion}")

        prev_emotion = self.current_emotion
        self.current_emotion = emotion
        self.last_emotion_change = time.time()

        # Clear previous expressions first
        if self.active_expressions:
            await self._clear_all_expressions()

        # If neutral, we're done (just cleared)
        if emotion == 'neutral':
            return

        # Get expressions for new emotion
        new_expressions = EMOTION_EXPRESSIONS.get(emotion, [])
        if not new_expressions:
            print(f"  No expressions defined for '{emotion}'")
            return

        # Pick random expression from available options
        available = [e for e in new_expressions if e in self.hotkey_ids]
        if not available:
            print(f"  No matched hotkeys for '{emotion}' expressions")
            return

        expression_name = random.choice(available)
        await self._trigger_expression(expression_name, emotion)

    async def trigger_by_name(self, expression_name: str):
        """
        Manually trigger an expression by internal name.
        Used for testing (e.g., keyboard button in motor).
        """
        if expression_name not in self.hotkey_ids:
            print(f"  Expression '{expression_name}' not matched to any VTS hotkey")
            return False

        # Clear any active expressions first
        if self.active_expressions:
            await self._clear_all_expressions()

        await self._trigger_expression(expression_name, "manual")
        return True

    async def _trigger_expression(self, expression_name: str, emotion: str):
        """Trigger a VTS expression hotkey by its real UUID."""
        hotkey_id = self.hotkey_ids.get(expression_name)
        if not hotkey_id:
            print(f"  No hotkey ID for: {expression_name}")
            return

        # Check if already active
        for active in self.active_expressions:
            if active.name == expression_name:
                print(f"  Expression '{expression_name}' already active, skipping")
                return

        success = await self.vts.trigger_hotkey(hotkey_id)

        if success:
            self.active_expressions.append(ActiveExpression(
                name=expression_name,
                hotkey_id=hotkey_id,
                triggered_at=time.time(),
                emotion=emotion
            ))
            print(f"  Triggered: {expression_name}")
        else:
            print(f"  Failed to trigger: {expression_name}")

    async def _clear_all_expressions(self):
        """Turn off all active expressions by re-triggering their hotkeys (toggle off)."""
        if not self.active_expressions:
            return

        print(f"  Clearing {len(self.active_expressions)} active expression(s)")

        for expr in self.active_expressions:
            await self.vts.trigger_hotkey(expr.hotkey_id)
            print(f"    Cleared: {expr.name}")

        self.active_expressions = []

    def get_available_expressions(self) -> List[str]:
        """Return list of expression names that have matched VTS hotkeys."""
        return list(self.hotkey_ids.keys())


if __name__ == '__main__':
    print("Expression Engine - Emotion → Expression Mappings:\n")

    for emotion, expressions in EMOTION_EXPRESSIONS.items():
        if expressions:
            expr_list = ', '.join(expressions)
            print(f"  {emotion:15s} → {expr_list}")
        else:
            print(f"  {emotion:15s} → (no expression)")

    print(f"\nTotal expressions defined: {len(EXPRESSION_DISPLAY_NAMES)}")
    print(f"Emotions with expressions: {sum(1 for e in EMOTION_EXPRESSIONS.values() if e)}")
    print(f"\nNote: Actual hotkey IDs are resolved at runtime from VTS")
