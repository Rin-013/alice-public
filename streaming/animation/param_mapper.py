#!/usr/bin/env python3
"""
Parameter mapping: Live2D clip parameters → VTS default parameters.

Clips contain Live2D model params (ParamAngleX, ParamEyeLOpen, etc.)
VTS expects VTS default params (FaceAngleX, EyeOpenLeft, etc.)

This mapper converts between them.
"""

from typing import Dict


# Mapping: Live2D param name → VTS default param name
LIVE2D_TO_VTS = {
    # Head angles
    "ParamAngleX": "FaceAngleX",
    "ParamAngleY": "FaceAngleY",
    "ParamAngleZ": "FaceAngleZ",

    # Eyes - open/close
    "ParamEyeLOpen": "EyeOpenLeft",
    "ParamEyeROpen": "EyeOpenRight",

    # Gaze (both eyes track together in VTS)
    "ParamEyeBallX": "EyeRightX",  # Map to right eye gaze
    "ParamEyeBallY": "EyeRightY",

    # Mouth
    "ParamMouthOpenY": "MouthOpen",
    "ParamMouthForm": "MouthSmile",
    "ParamMouthForm2": "MouthSmile",

    # Brows
    "ParamBrowLY": "BrowLeftY",
    "ParamBrowRY": "BrowRightY",

    # Body - REMOVED MAPPING
    # ParamBody* parameters can trigger toggles, so we DON'T map them
    # VTS will handle body movement through face tracking
}


def map_params(live2d_params: Dict[str, float]) -> Dict[str, float]:
    """
    Convert Live2D parameters to VTS default parameters.

    Args:
        live2d_params: Dict of {live2d_param_name: value}

    Returns:
        Dict of {vts_param_name: value}
    """
    vts_params = {}

    for live2d_name, value in live2d_params.items():
        # Check if we have a mapping
        if live2d_name in LIVE2D_TO_VTS:
            vts_name = LIVE2D_TO_VTS[live2d_name]
            vts_params[vts_name] = value
        # Keep params that are already VTS names
        elif live2d_name.startswith(('Face', 'Eye', 'Mouth', 'Brow')):
            vts_params[live2d_name] = value

    return vts_params


def get_core_animation_params(params: Dict[str, float]) -> Dict[str, float]:
    """
    Extract only the core animation parameters VTS cares about.
    Filters out obscure Live2D params and ANY toggle/accessory params.

    This prevents accidentally triggering toggles like flying head that were
    active when clips were recorded.

    Args:
        params: Any parameter dict

    Returns:
        Filtered dict with only essential animation params
    """
    # STRICT WHITELIST - only allow these exact parameter names
    # This prevents any toggle/accessory params from being sent
    core_vts_params = {
        'FaceAngleX', 'FaceAngleY', 'FaceAngleZ',
        'FacePositionX', 'FacePositionY', 'FacePositionZ',
        'EyeOpenLeft', 'EyeOpenRight',
        'EyeLeftX', 'EyeLeftY', 'EyeRightX', 'EyeRightY',
        # MouthOpen/MouthSmile excluded — VTS lip sync owns the mouth
        'BrowLeftY', 'BrowRightY',
    }

    # STRICT: Only return params that are in the whitelist
    # This blocks ALL generic Param* values that might be toggles
    filtered = {k: v for k, v in params.items() if k in core_vts_params}

    return filtered


def print_mapping_stats(live2d_params: Dict[str, float]):
    """Debug: print what params are being mapped."""
    mapped = []
    unmapped = []

    for name in live2d_params.keys():
        if name in LIVE2D_TO_VTS:
            vts_name = LIVE2D_TO_VTS[name]
            mapped.append(f"{name} → {vts_name}")
        elif name.startswith(('Face', 'Eye', 'Mouth', 'Brow')):
            mapped.append(f"{name} (already VTS)")
        else:
            unmapped.append(name)

    print(f"\n=== Parameter Mapping ===")
    print(f"Mapped: {len(mapped)}")
    for m in mapped[:10]:  # Show first 10
        print(f"  {m}")
    if len(mapped) > 10:
        print(f"  ... and {len(mapped) - 10} more")

    print(f"\nUnmapped (ignored): {len(unmapped)}")
    if unmapped:
        print(f"  {', '.join(unmapped[:5])}")
        if len(unmapped) > 5:
            print(f"  ... and {len(unmapped) - 5} more")


if __name__ == '__main__':
    # Test
    test_params = {
        'ParamAngleX': 10.0,
        'ParamAngleY': 5.0,
        'ParamEyeLOpen': 1.0,
        'ParamEyeROpen': 1.0,
        'ParamMouthOpenY': 0.5,
        'Param123': 0.0,  # Unknown param
    }

    print("Test input:")
    print(test_params)

    mapped = map_params(test_params)
    print("\nMapped output:")
    print(mapped)

    core = get_core_animation_params(mapped)
    print("\nCore params only:")
    print(core)
