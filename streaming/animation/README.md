# Animation System

Real-time avatar animation for Alice via VTube Studio.

**Last Updated:** June 2026
**Model:** Star Moon Jellyfish (Live2D)
**Hardware:** Consumer GPU

## Two engines (clip-based is production)

### Clip-based (`motion_engine.py`) — production path
Auto-launched by `chat.py`. Plays pre-recorded motion clips with emotion-based selection, smooth crossfade blending at 40 FPS.

### Procedural (`procedural_engine.py`) — built, needs calibration
Voice-reactive generative motion. Taps Alice's voice off VB-Cable "CABLE Output" loopback via `audio_listener.py` — RMS envelope + onsets drive head nod/lean/brow-pop. Idle drift persists when silent.

**Remaining:** calibrate `eye_open_rest` live (`[`/`]` keys, current default 0.70), smoke-test vs live VTS.

```bash
python streaming/animation/procedural_engine.py
python streaming/animation/procedural_engine.py --emotion excited
python streaming/animation/procedural_engine.py --audio-device "Microphone"
```

## Data flow

```
Mind emotion tag → emotion_state.json → Motion Engine (polls 100ms)
                                              │
                    ┌─────────┬───────────────┼────────────┐
                    ▼         ▼               ▼            ▼
              Movement   Expression       Mouth      Parameter
              Clips      Hotkeys          Control    Mapping
                    └─────────┴───────────────┴────────────┘
                                              ▼
                                    VTube Studio API → Live2D
```

## File structure

```
streaming/animation/
├── motion_engine.py         # Production — clip playback + expression + mouth
├── procedural_engine.py     # Alt — generative motion runner
├── procedural_motion.py     # Layered sine drift, darts, blinks, per-emotion profiles
├── audio_listener.py        # VB-Cable speech tap → envelope + onsets
├── expression_engine.py     # VTS hotkey expression manager
├── emotion_state.py         # Emotion state file API (JSON IPC)
├── param_mapper.py          # Live2D → VTS parameter mapping
├── easing.py                # Animation easing curves
├── capture_clip.py          # Clip recording tool
├── test_engines.py          # Interactive test suite
├── clips/                   # 128 motion clips (idle_*.json, expr_*.json)
├── tools/                   # Capture utilities, param mapping tools
```

## Quick start

```bash
# Clip-based (production)
python streaming/animation/motion_engine.py --emotion neutral

# Keyboard: 1-8 emotions, q quit
# Procedural adds: e onset pulse, [/] eye openness, --expressions flag
```

## Configuration

| Setting | Value | Notes |
|---|---|---|
| Playback speed | 0.9 (90%) | |
| FPS | 40 | |
| Smoothing | 0.8 | EMA filter |
| Clip duration | 8-30s | |
| Mouth dampening | 20% idle, 0% speaking | |
| Eye gaze | 70% | |

## VTS connection

WebSocket on `localhost:8001`. Uses `InjectParameterDataRequest` (40/sec) + `HotkeyTriggerRequest` (on emotion change). Strict 9-parameter whitelist (FaceAngle XYZ, EyeOpen L/R, EyeRight XY, MouthOpen, MouthSmile).

## Emotion → clip pools

Motion clips exist for: neutral, happy, sad, angry, excited, thinking, tired, surprised, bored, confused. 128 clips total (43 idle, rest expr). `talk_*` and `react_*` not yet recorded.
