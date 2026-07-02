#!/usr/bin/env python3
"""
Procedural motion - generative replacement for clip-based playback.

Produces continuous, emotion-conditioned VTS parameter frames without any
pre-recorded clip files. Every frame is computed from layered oscillators,
emotion-blended profiles, and procedural blinks/saccades.

Inputs per tick:
  - dt: seconds since last tick
  - state: current EmotionState (dominant emotion + accumulator scores)
  - audio_env: 0..1 normalized audio envelope (0 when silent)

Output: dict of VTS parameter values, fed straight through param_mapper.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from emotion_state import EmotionState


# ─── Emotion profiles ──────────────────────────────────────────────────
# Each profile shapes the motion: oscillation speed/amplitude, baseline
# lean, brow rest, eye openness, blink rate, gaze restlessness.

@dataclass
class EmotionProfile:
    speed: float = 1.0          # multiplier on all oscillation frequencies
    scale: float = 1.0          # multiplier on head/body amplitudes
    lean: float = 0.0           # baseline FacePositionZ (forward+ / back-)
    sway: float = 1.0           # lateral body sway amplitude multiplier
    brow_rest: float = 0.0      # baseline brow offset
    eye_open: float = 1.0       # baseline eye openness multiplier
    blink_hz: float = 0.35      # average blinks per second
    gaze_restless: float = 1.0  # saccade frequency multiplier


PROFILES: Dict[str, EmotionProfile] = {
    "neutral":   EmotionProfile(speed=1.10, scale=1.20),
    "happy":     EmotionProfile(speed=1.50, scale=1.60, lean=0.30, sway=1.2, brow_rest=0.20, gaze_restless=1.3),
    "excited":   EmotionProfile(speed=2.20, scale=2.10, lean=0.70, sway=1.6, brow_rest=0.40, blink_hz=0.55, gaze_restless=1.8),
    "sassy":     EmotionProfile(speed=1.40, scale=1.50, lean=0.25, sway=1.3, brow_rest=0.30, gaze_restless=1.5),
    "sad":       EmotionProfile(speed=0.65, scale=0.70, lean=-0.40, brow_rest=-0.30, eye_open=0.85, blink_hz=0.25),
    "angry":     EmotionProfile(speed=1.60, scale=1.60, lean=0.30, sway=1.2, brow_rest=-0.50, gaze_restless=1.4),
    "surprised": EmotionProfile(speed=1.80, scale=1.70, lean=0.20, brow_rest=0.70, eye_open=1.15, blink_hz=0.20, gaze_restless=1.6),
    "thinking":  EmotionProfile(speed=0.85, scale=1.00, lean=-0.15, brow_rest=-0.08, eye_open=0.95, gaze_restless=1.4),
    "tired":     EmotionProfile(speed=0.55, scale=0.65, lean=-0.50, brow_rest=-0.20, eye_open=0.65, blink_hz=0.45),
}


def blend_profiles(scores: Dict[str, float]) -> EmotionProfile:
    """Score-weighted blend of all profiles. Lets the existing emotion
    accumulator drive smooth profile transitions for free."""
    total = sum(scores.values())
    if total <= 1e-6:
        return PROFILES["neutral"]

    out = EmotionProfile(speed=0.0, scale=0.0, sway=0.0, eye_open=0.0, blink_hz=0.0, gaze_restless=0.0)
    for emo, weight in scores.items():
        p = PROFILES.get(emo, PROFILES["neutral"])
        w = weight / total
        out.speed += p.speed * w
        out.scale += p.scale * w
        out.lean += p.lean * w
        out.sway += p.sway * w
        out.brow_rest += p.brow_rest * w
        out.eye_open += p.eye_open * w
        out.blink_hz += p.blink_hz * w
        out.gaze_restless += p.gaze_restless * w
    return out


# ─── Layered oscillator ────────────────────────────────────────────────

@dataclass
class Oscillator:
    """Sum of N sines with prime-ish frequencies and random phase offsets.
    Smooth, never-repeating drift centered at zero with peak ~= sum(amps)."""
    freqs: Tuple[float, ...]
    amps: Tuple[float, ...]
    phases: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.phases:
            self.phases = [random.uniform(0, 2 * math.pi) for _ in self.freqs]

    def value(self, t: float, speed: float = 1.0) -> float:
        return sum(
            math.sin(t * 2 * math.pi * f * speed + p) * a
            for f, a, p in zip(self.freqs, self.amps, self.phases)
        )


# ─── Per-channel smoothing / velocity caps ────────────────────────────
# Higher smoothing = sluggish but stable. Eyes snap (low smoothing) so
# blinks and saccades land cleanly. Velocity caps are a teleport safety net.

PARAM_SMOOTHING: Dict[str, float] = {
    # Head: 0.78 - close to original 0.80. Drift + darts only now (no
    # bounce harmonics), so smoothing can stay high without killing motion.
    "FaceAngleX": 0.78, "FaceAngleY": 0.78, "FaceAngleZ": 0.80,
    "FacePositionX": 0.85, "FacePositionY": 0.88, "FacePositionZ": 0.84,
    "EyeOpenLeft": 0.35, "EyeOpenRight": 0.35,
    "EyeLeftX": 0.55, "EyeLeftY": 0.55, "EyeRightX": 0.55, "EyeRightY": 0.55,
    "BrowLeftY": 0.75, "BrowRightY": 0.75,
}
DEFAULT_SMOOTH = 0.76

PARAM_MAX_VEL: Dict[str, float] = {
    "FaceAngleX": 180.0, "FaceAngleY": 160.0, "FaceAngleZ": 140.0,
    "FacePositionX": 40.0, "FacePositionY": 40.0, "FacePositionZ": 35.0,
    "EyeOpenLeft": 12.0, "EyeOpenRight": 12.0,
    "EyeLeftX": 12.0, "EyeLeftY": 12.0, "EyeRightX": 12.0, "EyeRightY": 12.0,
    "BrowLeftY": 10.0, "BrowRightY": 10.0,
}
DEFAULT_MAX_VEL = 120.0


# ─── ProceduralMotion ──────────────────────────────────────────────────

class ProceduralMotion:
    """Generates continuous VTS parameter frames. No clip files."""

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        self.t = 0.0

        # Resting eyelid openness. On this model EyeOpenLeft=1.0 renders WIDE
        # (1.0 is the "surprised" extreme, not normal). Relaxed lids sit lower,
        # so all motion is anchored to this baseline and blinks pull it to ~0.
        # Tune live with [ / ] in procedural_engine.py, then bake the value here.
        self.eye_open_rest = 0.70

        # Slow drift layer - low-freq Perlin-ish wander (the "swaying").
        self.head_x = Oscillator(freqs=(0.04, 0.11, 0.27), amps=(5.5, 2.8, 1.4))
        self.head_y = Oscillator(freqs=(0.05, 0.13, 0.29), amps=(4.0, 2.0, 1.0))
        self.head_z = Oscillator(freqs=(0.03, 0.09, 0.21), amps=(3.5, 1.7, 0.8))

        # NOTE: the old constant 1-3Hz "bounce harmonics" are gone - sustained
        # mid-freq oscillation reads as shake on a 2D rig. The alive-while-
        # talking bounce now comes from speech motion (audio_bob/nod/roll),
        # which only runs when she's actually speaking. Darts cover idle.

        self.body_x = Oscillator(freqs=(0.025, 0.07),     amps=(1.8, 0.8))
        self.body_y = Oscillator(freqs=(0.020, 0.06),     amps=(1.0, 0.4))
        self.body_z = Oscillator(freqs=(0.06,  0.13),     amps=(0.6, 0.3))

        # Brows get a small faster harmonic so they don't feel frozen.
        self.brow_l = Oscillator(freqs=(0.07, 0.19, 0.8), amps=(0.22, 0.10, 0.04))
        self.brow_r = Oscillator(freqs=(0.07, 0.19, 0.8), amps=(0.22, 0.10, 0.04))

        # Speech motion - the lead actor when she's talking. A bob around
        # speaking cadence (~2Hz) plus a faster harmonic for syllable texture.
        # Bigger amps than the idle drift so her head clearly moves WITH her
        # voice (the Neuro "alive while talking" read), gated by audio_env.
        self.audio_nod = Oscillator(freqs=(1.9, 3.3), amps=(2.4, 1.1))
        self.audio_bob = Oscillator(freqs=(2.1, 4.3), amps=(1.4, 0.6))  # vertical bounce
        self.audio_roll = Oscillator(freqs=(0.9, 1.7), amps=(1.6, 0.7))  # head tilt sway

        # Head darts - discrete quick pivots between drift. This is the
        # Neuro-style "she just looked over" snap. Random target every
        # 2-5s, decays back over ~0.7s.
        self.next_dart_t = random.uniform(2.0, 4.0)
        self.dart_x = 0.0
        self.dart_y = 0.0
        self.dart_z = 0.0
        self.dart_decay = 1.4

        # Brow flicks - occasional discrete brow pops ("wait what?" moments).
        self.next_flick_t = random.uniform(3.0, 7.0)
        self.flick_intensity = 0.0
        self.flick_decay = 2.5

        # Blinks
        self.next_blink_t = random.uniform(1.0, 3.0)
        self.blink_progress = -1.0
        self.blink_duration = 0.13

        # Saccades
        self.next_saccade_t = random.uniform(2.0, 5.0)
        self.saccade_target_x = 0.0
        self.saccade_target_y = 0.0
        self.saccade_current_x = 0.0
        self.saccade_current_y = 0.0
        self.saccade_snap_speed = 14.0

        # Audio onset pulse (Phase 2 hook - brow pop + gaze flick on transients)
        self.onset_pulse = 0.0
        self.onset_decay_rate = 4.0

        self.prev_out: Dict[str, float] = {}

    def trigger_onset(self, intensity: float = 1.0) -> None:
        """Phase 2 hook: AudioListener calls this on detected transients."""
        self.onset_pulse = max(self.onset_pulse, min(1.0, intensity))

    def tick(self, dt: float, state: EmotionState, audio_env: float = 0.0) -> Dict[str, float]:
        self.t += dt
        profile = blend_profiles(state.scores)

        self.onset_pulse = max(0.0, self.onset_pulse - self.onset_decay_rate * dt)

        # Speech dominance: when she's talking, the random idle layer (drift +
        # darts) recedes so the voice-driven motion leads and reads as
        # intentional. In silence, idle_gain=1 and the wander carries her.
        speech = min(1.0, audio_env * 1.4)
        idle_gain = 1.0 - 0.55 * speech

        # Head: slow drift (long arc) + decaying darts (discrete pivots),
        # both pulled back while speaking. Speech motion is added below.
        head_x = self.head_x.value(self.t, profile.speed) * profile.scale * idle_gain
        head_y = self.head_y.value(self.t, profile.speed) * profile.scale * idle_gain
        head_z = self.head_z.value(self.t, profile.speed) * profile.scale * idle_gain

        # Dart scheduler - now the primary "alive" signal since bounce is off.
        # More frequent (1.5-3s) and slightly bigger to compensate.
        if self.t >= self.next_dart_t:
            self.dart_x = random.gauss(0, 3.5 * profile.scale)
            self.dart_y = random.gauss(0, 2.5 * profile.scale)
            self.dart_z = random.gauss(0, 1.8 * profile.scale)
            interval = random.uniform(1.5, 3.0) / max(0.5, profile.speed)
            self.next_dart_t = self.t + interval

        decay = math.exp(-self.dart_decay * dt)
        self.dart_x *= decay
        self.dart_y *= decay
        self.dart_z *= decay
        head_x += self.dart_x * idle_gain
        head_y += self.dart_y * idle_gain
        head_z += self.dart_z * idle_gain

        # Breathing - always-on subtle pitch oscillation
        head_y += math.sin(self.t * 2 * math.pi * 0.18) * 0.5

        # Speech motion - the lead while she talks. Nod (pitch), tilt (roll),
        # and a touch of yaw, all scaled by how loud/emphatic she is right now.
        # This is what makes the head move WITH her words instead of drifting.
        if audio_env > 0.02:
            amp = audio_env * profile.scale
            nod = self.audio_nod.value(self.t, profile.speed) * amp
            head_y += nod
            head_x += self.audio_nod.value(self.t * 0.5) * amp * 0.35
            head_z += self.audio_roll.value(self.t, profile.speed) * amp

        # Body - idle sway (pulled back while talking) + a vertical bounce and
        # forward lean driven by her voice. The bounce is the Neuro "bobbing
        # while she yaps" read; the lean makes her press in on emphasis.
        body_x = self.body_x.value(self.t, profile.speed) * profile.sway * idle_gain
        body_y = self.body_y.value(self.t, profile.speed) * 0.5 * idle_gain
        if audio_env > 0.02:
            body_y += self.audio_bob.value(self.t, profile.speed) * audio_env * profile.scale
        body_z = profile.lean + self.body_z.value(self.t, profile.speed) * 0.5 + audio_env * 1.1

        # Blink scheduler
        if self.t >= self.next_blink_t and self.blink_progress < 0:
            self.blink_progress = 0.0
            interval = max(0.5, 1.0 / max(profile.blink_hz, 0.05))
            self.next_blink_t = self.t + random.uniform(interval * 0.6, interval * 1.4)
            if random.random() < 0.10:  # occasional double-blink
                self.next_blink_t = self.t + self.blink_duration + 0.05

        if self.blink_progress >= 0:
            self.blink_progress += dt / self.blink_duration
            if self.blink_progress >= 1.0:
                self.blink_progress = -1.0
                blink_mod = 0.0
            else:
                blink_mod = math.sin(self.blink_progress * math.pi)
        else:
            blink_mod = 0.0
        eye_open = self.eye_open_rest * profile.eye_open * (1.0 - blink_mod)

        # Saccades
        if self.t >= self.next_saccade_t:
            self.saccade_target_x = max(-0.4, min(0.4, random.gauss(0, 0.18)))
            self.saccade_target_y = max(-0.2, min(0.2, random.gauss(0, 0.10)))
            self.next_saccade_t = self.t + random.uniform(2.0, 5.0) / max(0.3, profile.gaze_restless)

        snap = min(1.0, self.saccade_snap_speed * dt)
        self.saccade_current_x += (self.saccade_target_x - self.saccade_current_x) * snap
        self.saccade_current_y += (self.saccade_target_y - self.saccade_current_y) * snap
        gaze_x = self.saccade_current_x + self.onset_pulse * 0.05
        gaze_y = self.saccade_current_y + self.onset_pulse * 0.03

        # Brow flick scheduler - occasional discrete pop on both brows.
        if self.t >= self.next_flick_t:
            self.flick_intensity = random.uniform(0.4, 0.9) * profile.scale
            self.next_flick_t = self.t + random.uniform(3.0, 8.0) / max(0.5, profile.speed)
        self.flick_intensity = max(0.0, self.flick_intensity - self.flick_decay * dt)

        # Brows: rest + drift + flick + audio-onset pop. Slight L/R desync.
        brow_pop = self.onset_pulse * 0.25 + audio_env * 0.05 + self.flick_intensity
        brow_l = profile.brow_rest + self.brow_l.value(self.t, profile.speed) + brow_pop
        brow_r = profile.brow_rest + self.brow_r.value(self.t * 0.97, profile.speed) + brow_pop

        raw = {
            "FaceAngleX": head_x, "FaceAngleY": head_y, "FaceAngleZ": head_z,
            "FacePositionX": body_x, "FacePositionY": body_y, "FacePositionZ": body_z,
            "EyeOpenLeft": eye_open, "EyeOpenRight": eye_open,
            "EyeLeftX": gaze_x, "EyeLeftY": gaze_y,
            "EyeRightX": gaze_x, "EyeRightY": gaze_y,
            "BrowLeftY": brow_l, "BrowRightY": brow_r,
        }

        # Smooth + velocity-cap
        out: Dict[str, float] = {}
        for key, val in raw.items():
            prev = self.prev_out.get(key)
            if prev is None:
                out[key] = val
                continue
            s = PARAM_SMOOTHING.get(key, DEFAULT_SMOOTH)
            smoothed = s * prev + (1 - s) * val
            max_delta = PARAM_MAX_VEL.get(key, DEFAULT_MAX_VEL) * dt
            delta = smoothed - prev
            if abs(delta) > max_delta:
                smoothed = prev + math.copysign(max_delta, delta)
            out[key] = smoothed

        self.prev_out = dict(out)
        return out


if __name__ == "__main__":
    # Standalone smoke test - prints frames so you can sanity-check ranges.
    from emotion_state import EmotionState

    motion = ProceduralMotion(seed=42)
    state = EmotionState(emotion="excited", confidence=0.9,
                         scores={"excited": 0.8, "happy": 0.2})

    dt = 1.0 / 40
    print(f"80 frames @ 40fps ({80 * dt:.1f}s), emotion=excited:\n")
    for i in range(80):
        p = motion.tick(dt, state, audio_env=0.0)
        if i % 10 == 0:
            print(f"  t={i*dt:.2f}s  "
                  f"HX={p['FaceAngleX']:+.2f} HY={p['FaceAngleY']:+.2f} HZ={p['FaceAngleZ']:+.2f}  "
                  f"Eye={p['EyeOpenLeft']:.2f} Gaze={p['EyeLeftX']:+.2f}  "
                  f"BrowL={p['BrowLeftY']:+.2f} LeanZ={p['FacePositionZ']:+.2f}")
