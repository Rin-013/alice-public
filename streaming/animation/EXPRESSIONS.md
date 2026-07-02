# Alice Expression Hotkeys

## TODO: Expression Triggering System

**These expressions should be triggered/untriggered automatically in VTube Studio based on Alice's current emotional state.**

Implementation needed:
- Motion engine sends VTS hotkey commands (not just parameter injection)
- Map emotions to expressions (e.g., very happy → heart_eyes, embarrassed → blushing)
- Handle toggle on/off (trigger when emotion starts, untrigger when it ends)
- Expressions layer ON TOP of motion clips, not replacing them
- Consider intensity thresholds (only trigger heart_eyes at happiness > 0.8)

---

16 expressions for Alice to control. Use `capture_clip.py --session --category expr` to record them.

| Expression | Hotkey | Type |

## Accessories / Toggles
|------------|--------|------|
| Left Jellyfish | Left Shift + N7 | Accessory |
| Jellyfish | Left Ctrl + 4 | Accessory |
| Halo | Tab + 6 | Accessory |
| Shark Upper Teeth | Tab + 5 | Accessory |
| Hairclip | Tab + 4 | Accessory |
| Pearl Hairclip | Tab + 3 | Accessory |
| Shark Hairclip | Tab + 2 | Accessory |
| Moon Hairclip | Tab + 1 | Accessory |
| Shark Tail | Left Ctrl + 3 | Accessory |
| Flying Head | Left Shift + 1 | Pose |
| Shrink | Left Shift + 2 | Pose |


## Hide Toggles

| Expression | Hotkey | Type |
|------------|--------|------|
| Upper Teeth Hide | Left Ctrl + 8 | Hide |
| Back Skirt Hide | Left Ctrl + 7 | Hide |
| Messy Hair Hide | Left Ctrl + 6 | Hide |
| Animal Ear Hide | Left Ctrl + 5 | Hide |

## Poses / Gestures

| Expression | Hotkey | Type |
|------------|--------|------|
| Praying | Left Shift + 6 | Pose |
| Covering Chest | Left Shift + 5 | Pose |
| Heart Gesture | Left Shift + 4 | Pose |
| Holding Star | Left Shift + 3 | Pose |

## Eye Expressions

| Expression | Hotkey | Type |
|------------|--------|------|
| White Eyes | Left Ctrl + 2 | Eyes |
| Black Eyes | Left Ctrl + 1 | Eyes |
| Heart Eyes | Right Shift + 9 | Eyes |
| Star Eyes | Right Shift + 8 | Eyes |
| Dizzy | Right Shift + 6 | Eyes |

## Face Expressions

| Expression | Hotkey | Type |
|------------|--------|------|
| Crying | Right Shift + 7 | Face |
| Blushing | Right Shift + 5 | Face |
| Dark Face | Right Shift + 4 | Face |
| ZZZ (Sleepy) | Right Shift + 3 | Face |
| Speechless | Right Shift + 2 | Face |
| Angry | Right Shift + 1 | Face |
| Tongue Out | Left Ctrl + 0 | Face |

---

## Quick Reference (by hotkey)

### Tab + Number
- Tab 1: Moon Hairclip
- Tab 2: Shark Hairclip
- Tab 3: Pearl Hairclip
- Tab 4: Hairclip
- Tab 5: Shark Upper Teeth
- Tab 6: Halo

### Left Shift + Number
- LShift 1: Flying Head
- LShift 2: Shrink
- LShift 3: Holding Star
- LShift 4: Heart Gesture
- LShift 5: Covering Chest
- LShift 6: Praying
- LShift N7: Left Jellyfish

### Left Ctrl + Number
- LCtrl 0: Tongue Out
- LCtrl 1: Black Eyes
- LCtrl 2: White Eyes
- LCtrl 3: Shark Tail
- LCtrl 4: Jellyfish
- LCtrl 5: Animal Ear Hide
- LCtrl 6: Messy Hair Hide
- LCtrl 7: Back Skirt Hide
- LCtrl 8: Upper Teeth Hide

### Right Shift + Number
- RShift 1: Angry
- RShift 2: Speechless
- RShift 3: ZZZ (Sleepy)
- RShift 4: Dark Face
- RShift 5: Blushing
- RShift 6: Dizzy
- RShift 7: Crying
- RShift 8: Star Eyes
- RShift 9: Heart Eyes
