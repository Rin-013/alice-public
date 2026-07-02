"""
Complete ARKit Blendshape to VTube Studio parameter mapping.

Based on Star Moon Jellyfish model parameters from VTS screenshots.

IMPORTANT: The trained model uses Sigmoid (outputs 0-1), but head rotation
in training data has negative values. The model learns ~0.5 = neutral.
Transform functions account for this.

VTS Input Ranges (from screenshots):
- FaceAngleX/Y/Z: -25 to 25 degrees input → -30 to 30 degrees output
- EyeOpenLeft/Right: 0 to 0.5 input → 0 to 1.2 output
- MouthOpen: 0 to 0.6 input → 0 to 1 output
- MouthSmile: 0.21 to 1 input → 0 to 1 output
"""

# ARKit blendshape name → (VTS input parameter, transform function)
# Transform converts model output (0-1 from Sigmoid) to VTS expected range

ARKIT_TO_VTS_COMPLETE = {
    # ==================== HEAD ====================
    # Model outputs 0-1, we interpret 0.5 as neutral
    # Scale to VTS range: -25 to 25 degrees
    'HeadYaw': ('FaceAngleY', lambda x: (x - 0.5) * 50),      # 0.5→0, 0→-25, 1→+25
    'HeadPitch': ('FaceAngleX', lambda x: -(x - 0.5) * 50),   # Inverted for VTS
    'HeadRoll': ('FaceAngleZ', lambda x: (x - 0.5) * 50),

    # ==================== EYES - OPEN/CLOSE ====================
    # Model output: 0=blink (closed), 1=open
    # VTS EyeOpenLeft/Right: 0-0.5 input range
    # Note: VTS has eyes swapped - EyeOpenLeft controls right eye display
    'EyeBlinkLeft': ('EyeOpenRight', lambda x: (1.0 - x) * 0.5),   # Invert + scale
    'EyeBlinkRight': ('EyeOpenLeft', lambda x: (1.0 - x) * 0.5),

    # Wide eyes add to openness
    'EyeWideLeft': ('EyeOpenRight', lambda x: x * 0.15),
    'EyeWideRight': ('EyeOpenLeft', lambda x: x * 0.15),

    # Squint reduces openness
    'EyeSquintLeft': ('EyeOpenRight', lambda x: -x * 0.1),
    'EyeSquintRight': ('EyeOpenLeft', lambda x: -x * 0.1),

    # ==================== EYES - GAZE ====================
    # Eye rotation: model outputs 0-1, interpret 0.5 as center
    'EyeLookInLeft': ('EyeRightX', lambda x: x * 0.3),
    'EyeLookOutLeft': ('EyeRightX', lambda x: -x * 0.3),
    'EyeLookUpLeft': ('EyeRightY', lambda x: x * 0.3),
    'EyeLookDownLeft': ('EyeRightY', lambda x: -x * 0.3),

    # Direct eye rotation (radians in training data, 0.5=center in model)
    'LeftEyeYaw': ('EyeRightX', lambda x: (x - 0.5) * 1.0),
    'LeftEyePitch': ('EyeRightY', lambda x: -(x - 0.5) * 1.0),

    # ==================== MOUTH - OPEN/CLOSE ====================
    # VTS MouthOpen expects 0-0.6 range
    'JawOpen': ('MouthOpen', lambda x: x * 0.6),
    'MouthClose': ('MouthOpen', lambda x: -x * 0.2),

    # Chew animation uses JawOpen input
    'JawForward': ('JawOpen', lambda x: x * 0.3),

    # Crooked mouth (MouthX)
    'JawLeft': ('MouthX', lambda x: -x * 0.5),
    'JawRight': ('MouthX', lambda x: x * 0.5),

    # ==================== MOUTH - SHAPE ====================
    # VTS MouthSmile: 0.21-1 input range
    'MouthSmileLeft': ('MouthSmile', lambda x: 0.21 + x * 0.79),
    'MouthSmileRight': ('MouthSmile', lambda x: 0.21 + x * 0.79),

    # Frown reduces smile
    'MouthFrownLeft': ('MouthSmile', lambda x: -x * 0.5),
    'MouthFrownRight': ('MouthSmile', lambda x: -x * 0.5),

    # Funnel (O shape) - MouthFunnel 0-1
    'MouthFunnel': ('MouthFunnel', lambda x: x),

    # Pucker - MouthPucker 0-0.3 range from screenshot
    'MouthPucker': ('MouthPucker', lambda x: x * 0.3),

    # Mouth X position (crooked)
    'MouthLeft': ('MouthX', lambda x: -x * 0.3),
    'MouthRight': ('MouthX', lambda x: x * 0.3),

    # Stretch contributes to smile
    'MouthStretchLeft': ('MouthSmile', lambda x: x * 0.3),
    'MouthStretchRight': ('MouthSmile', lambda x: x * 0.3),

    # Dimples
    'MouthDimpleLeft': ('MouthSmile', lambda x: x * 0.2),
    'MouthDimpleRight': ('MouthSmile', lambda x: x * 0.2),

    # Lip roll
    'MouthRollLower': ('MouthOpen', lambda x: -x * 0.1),
    'MouthRollUpper': ('MouthOpen', lambda x: -x * 0.1),

    # Shrug/pout - MouthShrug 0-0.8 range
    'MouthShrugLower': ('MouthShrug', lambda x: x * 0.8),
    'MouthShrugUpper': ('MouthShrug', lambda x: x * 0.8),

    # Press - MouthPressLipOpen
    'MouthPressLeft': ('MouthPressLipOpen', lambda x: x * 0.5),
    'MouthPressRight': ('MouthPressLipOpen', lambda x: x * 0.5),

    # Lower/Upper lip movement
    'MouthLowerDownLeft': ('MouthOpen', lambda x: x * 0.3),
    'MouthLowerDownRight': ('MouthOpen', lambda x: x * 0.3),
    'MouthUpperUpLeft': ('MouthOpen', lambda x: x * 0.2),
    'MouthUpperUpRight': ('MouthOpen', lambda x: x * 0.2),

    # ==================== BROWS ====================
    # BrowLeftY: negative = down, positive = up
    'BrowDownLeft': ('BrowLeftY', lambda x: -x * 0.5),
    'BrowDownRight': ('BrowRightY', lambda x: -x * 0.5),
    'BrowInnerUp': ('BrowLeftY', lambda x: x * 0.4),
    'BrowOuterUpLeft': ('BrowLeftY', lambda x: x * 0.3),
    'BrowOuterUpRight': ('BrowRightY', lambda x: x * 0.3),

    # ==================== CHEEKS ====================
    'CheekPuff': ('CheekPuff', lambda x: x),
    'CheekSquintLeft': ('CheekPuff', lambda x: x * 0.1),
    'CheekSquintRight': ('CheekPuff', lambda x: x * 0.1),

    # ==================== NOSE ====================
    'NoseSneerLeft': ('BrowLeftY', lambda x: -x * 0.1),
    'NoseSneerRight': ('BrowRightY', lambda x: -x * 0.1),

    # ==================== TONGUE ====================
    'TongueOut': ('TongueOut', lambda x: x),
}


def convert_blendshapes_to_vts(blendshapes: dict, mapping: dict = None) -> dict:
    """
    Convert ARKit blendshapes to VTS parameters.

    Args:
        blendshapes: Dict of {arkit_name: value}
        mapping: Optional custom mapping dict

    Returns:
        Dict of {vts_param: value}
    """
    if mapping is None:
        mapping = ARKIT_TO_VTS_COMPLETE

    vts_params = {}
    param_counts = {}  # Track how many sources contribute to each param

    for arkit_name, value in blendshapes.items():
        if arkit_name not in mapping:
            continue

        vts_name, transform = mapping[arkit_name]
        transformed_value = transform(value)

        # Accumulate values for averaging
        if vts_name in vts_params:
            vts_params[vts_name] += transformed_value
            param_counts[vts_name] += 1
        else:
            vts_params[vts_name] = transformed_value
            param_counts[vts_name] = 1

    # Average parameters that had multiple sources
    for name in vts_params:
        if param_counts[name] > 1:
            vts_params[name] /= param_counts[name]

    return vts_params


# Simplified mapping for quick testing (only essential params)
# Uses same transforms as complete mapping
ARKIT_TO_VTS_SIMPLE = {
    'HeadYaw': ('FaceAngleY', lambda x: (x - 0.5) * 50),
    'HeadPitch': ('FaceAngleX', lambda x: -(x - 0.5) * 50),
    'HeadRoll': ('FaceAngleZ', lambda x: (x - 0.5) * 50),
    'EyeBlinkLeft': ('EyeOpenRight', lambda x: (1.0 - x) * 0.5),
    'EyeBlinkRight': ('EyeOpenLeft', lambda x: (1.0 - x) * 0.5),
    'JawOpen': ('MouthOpen', lambda x: x * 0.6),
    'MouthSmileLeft': ('MouthSmile', lambda x: 0.21 + x * 0.79),
    'MouthSmileRight': ('MouthSmile', lambda x: 0.21 + x * 0.79),
}
