#!/usr/bin/env python3
"""
Easing functions for smooth animation transitions.

Based on Robert Penner's easing equations and CSS easing standards.
See: https://easings.net/
"""

import math


def linear(t: float) -> float:
    """Linear easing (no acceleration)."""
    return t


def ease_in_out_cubic(t: float) -> float:
    """
    Cubic ease in-out (recommended for most transitions).
    Starts slow, speeds up in middle, ends slow.

    CSS equivalent: cubic-bezier(0.65, 0, 0.35, 1)
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_out_quad(t: float) -> float:
    """
    Quadratic ease in-out (smoother than cubic).
    Good for gentle movements.

    CSS equivalent: cubic-bezier(0.45, 0, 0.55, 1)
    """
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - pow(-2 * t + 2, 2) / 2


def ease_in_out_sine(t: float) -> float:
    """
    Sinusoidal ease in-out (very smooth).
    Best for natural, organic motion.

    CSS equivalent: cubic-bezier(0.37, 0, 0.63, 1)
    """
    return -(math.cos(math.pi * t) - 1) / 2


def ease_out_expo(t: float) -> float:
    """
    Exponential ease out (starts fast, slows dramatically).
    Good for objects coming to rest.
    """
    return 1 if t == 1 else 1 - pow(2, -10 * t)


def ease_in_cubic(t: float) -> float:
    """Cubic ease in - starts slow, accelerates."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease out - starts fast, decelerates."""
    return 1 - pow(1 - t, 3)


# Recommended easing function for avatar motion
DEFAULT_EASING = ease_in_out_sine  # Most natural for character animation


if __name__ == '__main__':
    # Visualize easing curves
    import numpy as np

    print("Easing Function Comparison (0.0 to 1.0):\n")

    easings = {
        'Linear': linear,
        'EaseInOutCubic': ease_in_out_cubic,
        'EaseInOutQuad': ease_in_out_quad,
        'EaseInOutSine': ease_in_out_sine,
    }

    t_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"{'t':<6}", end='')
    for name in easings.keys():
        print(f"{name:<18}", end='')
    print()
    print("-" * 80)

    for t in t_values:
        print(f"{t:<6.1f}", end='')
        for easing_func in easings.values():
            result = easing_func(t)
            print(f"{result:<18.4f}", end='')
        print()

    print("\nRecommended: ease_in_out_sine for natural character motion")
