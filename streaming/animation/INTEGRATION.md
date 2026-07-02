# Motion Engine Integration Guide

## 3-Part Animation System

### 1. Movement Engine (Motion Clips)
**File:** `motion_engine.py`
- Plays motion clips (idle, talk, react) based on current emotion
- Clips auto-select from emotion groups
- Runs at 40 FPS with smooth blending

### 2. Expression Engine (VTS Hotkeys)
**File:** `expression_engine.py`
- Triggers VTS expression hotkeys based on emotion
- Expressions layer on top of motion clips
- Auto-clears when emotion returns to neutral

### 3. Mouth/Speech Engine (TTS Integration)
**Status:** Scaffolding in place
- When `is_speaking=True`, motion clips do NOT control mouth
- VTS audio detection will handle lip sync from TTS audio
- Need to route TTS output to VTS as microphone input

---

## EmotionBERT Integration

### How to Send Emotions to Motion Engine

EmotionBERT should write to `emotion_state.json` whenever emotion changes:

```python
from motor.animation.emotion_state import set_emotion

# When EmotionBERT detects new emotion
detected_emotion = "happy"  # or sad, angry, flirty, etc.
confidence = 0.85

set_emotion(detected_emotion, confidence=confidence)
```

### When Alice is Speaking

When TTS starts/stops:

```python
from motor.animation.emotion_state import set_speaking

# When TTS starts
set_speaking(True)

# When TTS finishes
set_speaking(False)
```

Or combined:

```python
from motor.animation.emotion_state import set_emotion

# Emotion with speaking flag
set_emotion("excited", confidence=0.9, is_speaking=True)
```

---

## Emotion → Expression Mapping

Current mappings (from `expression_engine.py`):

| Emotion | Expressions | Notes |
|---------|-------------|-------|
| `flirty` | heart_eyes, heart_gesture, tongue_out | Teasing/flirting |
| `excited` | star_eyes | Enthusiasm |
| `sad` | crying | Sadness |
| `tired` | zzz_sleepy | Exhaustion |
| `angry` | angry | Also for pouting |
| `disappointed` | dark_face | Disappointment |
| `surprised` | speechless, dizzy | Shock |
| `embarrassed` | blushing | Flustered |
| `happy` | (none) | Just smiling motion |
| `neutral` | (none) | Clears all expressions |

### Adding New Mappings

Edit `expression_engine.py` and update `EMOTION_EXPRESSIONS` dict:

```python
EMOTION_EXPRESSIONS = {
    'new_emotion': ['expression_name1', 'expression_name2'],
    # ...
}
```

---

## Running the Complete System

### Start Motion Engine

```bash
python3 motor/animation/motion_engine.py --emotion neutral
```

This runs all 3 engines:
- ✓ Motion clips (emotion-based)
- ✓ Expressions (emotion-triggered)
- ✓ Mouth control (TTS-aware)

### Manual Testing

Keyboard controls (while running):
- `1-8`: Change emotion manually
- `q`: Quit

Or write to emotion state file:

```bash
python3 -c "from motor.animation.emotion_state import set_emotion; set_emotion('happy', 0.9)"
```

---

## Architecture Flow

```
EmotionBERT Output
       ↓
emotion_state.json ← set_emotion()
       ↓
Motion Engine reads every 100ms
       ↓
┌──────┴──────┬──────────────┬─────────────┐
│             │              │             │
Movement    Expression    Mouth         │
Engine      Engine        Control       │
│             │              │             │
Plays       Triggers      Zeros out     │
motion      VTS           mouth if      │
clips       hotkeys       speaking      │
│             │              │             │
└──────┬──────┴──────────────┴─────────────┘
       ↓
VTS Parameter Injection + Hotkey Triggers
       ↓
VTube Studio Avatar Animation
```

---

## TTS Integration (TODO)

### What's Needed:

1. **Virtual Audio Cable** (macOS: BlackHole, Windows: VB-Audio)
   - Route TTS output → Virtual cable
   - VTS listens to virtual cable for lip sync

2. **TTS Integration Points:**
   - Before TTS starts: `set_speaking(True)`
   - After TTS ends: `set_speaking(False)`
   - Motion engine will automatically zero out mouth params when speaking

3. **VTS Configuration:**
   - Settings → Audio
   - Select virtual audio cable as input
   - Enable lip sync from microphone

---

## Troubleshooting

### Expressions not triggering
- Check VTS hotkeys are configured correctly (see `EXPRESSIONS.md`)
- Verify hotkey strings match in `expression_engine.py` → `VTS_HOTKEYS`
- Check VTS API is enabled and accepting connections

### Flying head appearing
- Should be fixed - `expr_*` clips are blocked from motion playback
- Expressions only trigger via hotkeys, not clip playback

### Emotions not changing
- Check `emotion_state.json` is being written
- Verify motion engine is reading emotion state (should print emotion changes)
- Motion engine polls every 100ms

### Mouth moving during speech
- Ensure `set_speaking(True)` is called when TTS starts
- Motion engine zeros out mouth params when `is_speaking=True`
- VTS audio lip sync takes over

---

## Files Reference

| File | Purpose |
|------|---------|
| `motion_engine.py` | Main engine - runs all 3 systems |
| `expression_engine.py` | VTS hotkey expression triggering |
| `emotion_state.py` | Emotion state file read/write |
| `param_mapper.py` | Live2D → VTS parameter mapping |
| `capture_clip.py` | Record new motion clips |
| `EXPRESSIONS.md` | VTS hotkey reference |
| `INTEGRATION.md` | This file |
