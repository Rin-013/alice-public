"""
Star Moon Jellyfish model: Live2D parameter → VTS Input parameter mapping.

Auto-generated from star_moon_jellyfish.vtube.json ParameterSettings.

When we record with iPhone tracking, we get Live2D OUTPUT values.
To inject back into VTS, we need INPUT values with proper scaling.

Each entry: Live2D_param → (VTS_input_param, conversion_function)
The conversion function transforms output values back to input values.
"""

# Live2D param → (VTS input param, output→input conversion)
LIVE2D_TO_INPUT_WITH_SCALE = {
    # Brows
    'ParamBrowRY2': ('BrowLeftY', lambda v: 0.4 + (v - (-1.0)) * 0.6 / 2.0),

    # Mouth
    'ParamMouthForm2': ('MouthSmile', lambda v: 0.3 + (v - (-1.0)) * 0.5 / 2.0),
    'ParamMouthOpenY': ('MouthOpen', lambda v: 0.05 + (v - 0.0) * 0.55 / 1.0),
    'ParamMouthForm': ('MouthPucker', lambda v: -0.3 + (v - (-1.0)) * 0.6 / 2.0),
    'Param139': ('MouthX', lambda v: -1.0 + (v - (-1.0)) * 2.0 / 2.0),  # 1:1 scale
    'Param147': ('JawOpen', lambda v: v),  # 1:1 scale
    'Param148': ('MouthShrug', lambda v: v * 0.8),
    'Param155': ('MouthFunnel', lambda v: v),  # 1:1 scale
    'Param160': ('MouthPressLipOpen', lambda v: -0.5 + (v - (-1.0)) * 1.0 / 2.0),

    # Eyes - smile (driven by MouthSmile input)
    'ParamEyeLSmile': ('MouthSmile', lambda v: 0.3 + v * 0.7),
    'ParamEyeRSmile': ('MouthSmile', lambda v: 0.3 + v * 0.7),

    # Eyes - open (note: swapped, and scaled from 0-1.2 output to 0-0.5 input)
    'ParamEyeROpen': ('EyeOpenLeft', lambda v: v * 0.5 / 1.2),   # output 0-1.2 → input 0-0.5
    'ParamEyeLOpen': ('EyeOpenRight', lambda v: v * 0.5 / 1.2),  # output 0-1.2 → input 0-0.5

    # Eyes - gaze (inverted X axis)
    'ParamEyeBallX': ('EyeRightX', lambda v: -0.5 + (v - 1.0) * 1.0 / (-2.0)),
    'ParamEyeBallY': ('EyeRightY', lambda v: -1.0 + (v - (-0.5)) * 2.0 / 1.0),

    # Face angles (output -30 to 30 → input -25 to 25)
    'ParamAngleX2': ('FaceAngleX', lambda v: -25.0 + (v - (-30.0)) * 50.0 / 60.0),
    'ParamAngleX3': ('FaceAngleY', lambda v: -25.0 + (v - (-30.0)) * 50.0 / 60.0),
    'ParamAngleX4': ('FaceAngleZ', lambda v: -25.0 + (v - (-30.0)) * 50.0 / 60.0),

    # Cheeks
    'Param128': ('CheekPuff', lambda v: v),  # 1:1 scale

    # Tongue
    'fase85': ('TongueOut', lambda v: v),  # 1:1 scale
}

# Simple name-only mapping (no scaling) for reference
LIVE2D_TO_INPUT = {k: v[0] for k, v in LIVE2D_TO_INPUT_WITH_SCALE.items()}


def convert_live2d_to_input(live2d_params: dict) -> dict:
    """
    Convert Live2D parameter values to VTS input parameters with proper scaling.

    Args:
        live2d_params: Dict of {live2d_param_name: value}

    Returns:
        Dict of {input_param_name: value}
    """
    input_params = {}
    param_counts = {}

    for live2d_name, value in live2d_params.items():
        if live2d_name in LIVE2D_TO_INPUT_WITH_SCALE:
            input_name, convert_fn = LIVE2D_TO_INPUT_WITH_SCALE[live2d_name]
            converted_value = convert_fn(value)

            # Average if multiple Live2D params map to same input
            if input_name in input_params:
                input_params[input_name] += converted_value
                param_counts[input_name] += 1
            else:
                input_params[input_name] = converted_value
                param_counts[input_name] = 1

    # Average the values
    for name in input_params:
        if param_counts[name] > 1:
            input_params[name] /= param_counts[name]

    return input_params


def get_mapped_param_names(live2d_names: list) -> list:
    """Get the input parameter names for a list of Live2D names."""
    input_names = set()
    unmapped = []
    for name in live2d_names:
        if name in LIVE2D_TO_INPUT:
            input_names.add(LIVE2D_TO_INPUT[name])
        else:
            unmapped.append(name)

    if unmapped:
        print(f"Warning: {len(unmapped)} Live2D params have no input mapping")

    return sorted(list(input_names))
