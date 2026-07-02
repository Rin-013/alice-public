# Star Moon Jellyfish UI — stream overlay reference assets

Rendered overlay assets in the kawaii blue / jellyfish aesthetic.
Sourced as design reference for Alice's stream layout. Original folder
name was `星月水母UI` ("Star Moon Jellyfish UI").

These are static PNGs — not editable layered sources. Use them as the
visual target when building Alice's actual OBS scene composition.

## Files

| File | What it is | Use for |
|---|---|---|
| `main_layout.png` | Big content panel (game/screen capture) + chat/comments panel right + title/search bar bottom | Main "in-game" or "just chatting" OBS scene |
| `camera_layout.png` | Comments panel left (with circular camera/avatar slot) + Target panel right + chat box | Face-cam-focused / talking-head OBS scene |
| `live_intro.png` | "配信中 / LIVE" full-screen card with halo'd chibi + jellyfish mascot | Starting-soon / stream intro scene |
| `live_intro_alt.png` | Variant of the above (slight composition differences) | Alt starting-soon / outro scene |
| `chibi_mascot.png` | Halo'd chibi girl with floating jellyfish, eyes closed | Subscriber/donation alert overlay, channel emote, watermark |

## Aesthetic constraints to match in editable rebuild

- Pastel blue (#B8D4F0 ish) + white gingham/check pattern
- Lace / doily borders on panels
- Cinnamoroll-adjacent jellyfish + halo'd chibi mascots
- Star + cloud accents
- Soft glow / sparkle decorations
- All-caps Latin "LIVE" alongside Chinese 配信中 — for an English stream
  the rebuild should drop the Chinese and keep "LIVE"

## Status

Reference only — Alice's actual production overlays still need to be
either:
- (a) used as-is (accept the foreign branding)
- (b) rebuilt as editable Figma/PSD sources matching this aesthetic
- (c) commissioned from an overlay artist (Etsy / Fiverr) using
  these as the brief

See parent `motor/stream_overlays/` for any future rebuild folders.
