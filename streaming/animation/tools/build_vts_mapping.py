#!/usr/bin/env python3
"""
Build VTS parameter mapping from star_moon_jellyfish.vtube.json.

This extracts the input/output ranges and creates conversion functions
to transform Live2D output values back to VTS input values.
"""

import json
from pathlib import Path


def build_mapping(vtube_json_path: str) -> dict:
    """
    Build mapping from vtube.json ParameterSettings.

    Returns dict with:
        live2d_name: {
            'input_name': str,
            'input_range': (lower, upper),
            'output_range': (lower, upper),
            'convert': function to convert output→input
        }
    """
    with open(vtube_json_path) as f:
        data = json.load(f)

    mapping = {}

    for param in data.get('ParameterSettings', []):
        input_name = param.get('Input', '')
        output_name = param.get('OutputLive2D', '')

        if not input_name or not output_name:
            continue

        in_lo = param.get('InputRangeLower', 0)
        in_hi = param.get('InputRangeUpper', 1)
        out_lo = param.get('OutputRangeLower', 0)
        out_hi = param.get('OutputRangeUpper', 1)

        # Create conversion function: output → input
        # input = in_lo + (output - out_lo) * (in_hi - in_lo) / (out_hi - out_lo)
        out_range = out_hi - out_lo
        in_range = in_hi - in_lo

        if abs(out_range) < 0.0001:
            # No output range, just use input midpoint
            def convert(v, in_lo=in_lo, in_hi=in_hi):
                return (in_lo + in_hi) / 2
        else:
            def convert(v, in_lo=in_lo, in_range=in_range, out_lo=out_lo, out_range=out_range):
                return in_lo + (v - out_lo) * in_range / out_range

        mapping[output_name] = {
            'input_name': input_name,
            'input_range': (in_lo, in_hi),
            'output_range': (out_lo, out_hi),
            'convert': convert
        }

    return mapping


def generate_mapping_code(vtube_json_path: str):
    """Generate Python code for the mapping."""
    with open(vtube_json_path) as f:
        data = json.load(f)

    print("# Auto-generated from", vtube_json_path)
    print("# Live2D param → (VTS input param, conversion function)")
    print()
    print("LIVE2D_TO_INPUT_WITH_SCALE = {")

    for param in data.get('ParameterSettings', []):
        input_name = param.get('Input', '')
        output_name = param.get('OutputLive2D', '')

        if not input_name or not output_name:
            continue

        in_lo = param.get('InputRangeLower', 0)
        in_hi = param.get('InputRangeUpper', 1)
        out_lo = param.get('OutputRangeLower', 0)
        out_hi = param.get('OutputRangeUpper', 1)

        out_range = out_hi - out_lo
        in_range = in_hi - in_lo

        # Generate conversion lambda
        if abs(out_range) < 0.0001:
            conv = f"lambda v: {(in_lo + in_hi) / 2}"
        else:
            # input = in_lo + (v - out_lo) * in_range / out_range
            conv = f"lambda v: {in_lo} + (v - {out_lo}) * {in_range} / {out_range}"

        print(f"    '{output_name}': ('{input_name}', {conv}),")

    print("}")


if __name__ == '__main__':
    vtube_path = Path(__file__).parent.parent / 'star_moon_jelly' / 'star_moon_jellyfish.vtube.json'
    generate_mapping_code(vtube_path)
