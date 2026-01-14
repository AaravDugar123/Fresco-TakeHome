"""Geometry-based door classification rules."""
import numpy as np
from typing import List, Dict, Optional
from src.geometry_analyzer import get_bezier_radius


def calculate_arc_sweep_angle(arc: Dict, radius: float) -> Optional[float]:
    """
    Calculates the sweep angle (in degrees) using the Chord/Radius ratio.
    
    Args:
        arc: Arc dictionary with control_points
        radius: Pre-calculated radius (to avoid redundant calculation)
    """
    pts = arc['control_points']
    if len(pts) != 4:
        return None

    if radius <= 0:
        return None
    
    # Calculate Chord Length (Straight line distance start to end)
    p0 = np.array(pts[0])  # Start
    p3 = np.array(pts[3])  # End
    chord_len = np.linalg.norm(p3 - p0)
    
    # Safety Check: A chord cannot be longer than the Diameter (2*R)
    if chord_len > 2 * radius:
        chord_len = 2 * radius  # Cap it to avoid math domain errors

    # Calculate Angle
    # Formula: Angle = 2 * arcsin( (Chord/2) / Radius )
    angle_radians = 2 * np.arcsin(chord_len / (2 * radius))
    
    return np.degrees(angle_radians)


def check_arc_line_touch(arc: Dict, line: Dict, arc_radius: float) -> bool:
    """
    Check if arc and line touch at one endpoint.

    One endpoint of the line should be very close to one endpoint of the arc.
    "Very close" = within 10% of the arc radius.

    Args:
        arc: Arc dictionary with control_points
        line: Line dictionary with start and end
        arc_radius: Radius of the arc

    Returns:
        True if arc and line touch at an endpoint
    """
    arc_start = np.array(arc['control_points'][0])
    arc_end = np.array(arc['control_points'][-1])
    line_start = np.array(line['start'])
    line_end = np.array(line['end'])

    threshold = arc_radius * .2

    distances = [
        np.linalg.norm(arc_start - line_start),
        np.linalg.norm(arc_start - line_end),
        np.linalg.norm(arc_end - line_start),
        np.linalg.norm(arc_end - line_end)
    ]

    return min(distances) <= threshold


def classify_swing_door(arc: Dict, line: Dict) -> Optional[Dict]:
    result = get_bezier_radius(arc['control_points'])
    if result is None:
        return None
    arc_radius, arc_center = result

    # Rule 1: Arc sweep check (70-120 degrees) - pass radius to avoid recalculation
    sweep_angle = calculate_arc_sweep_angle(arc, arc_radius)
    if sweep_angle is None or not (70 <= sweep_angle <= 120):
        return None

    # Rule 2: Line length ≈ arc radius (ratio-based)
    line_length = np.linalg.norm(np.array(line['end']) - np.array(line['start']))
    ratio = line_length / arc_radius
    if not (0.6 < ratio < 1.4):  # Within 40%
        return None

    # Rule 3: Arc and line touch check
    touch_result = check_arc_line_touch(arc, line, arc_radius)
    if not touch_result:
        return None

    # All rules passed - this is a swing door
    return {
        "type": "swing_door",
        "arc": arc,
        "line": line,
        "arc_radius": arc_radius,
        "sweep_angle": sweep_angle,
        "center": arc_center
    }


def classify_swing_doors(arcs: List[Dict], lines: List[Dict]) -> List[Dict]:
    """
    Classify all swing doors from arcs and lines.

    Args:
        arcs: List of arc dictionaries
        lines: List of line dictionaries

    Returns:
        List of classified swing door dictionaries
    """
    swing_doors = []
    used_lines = set()
    used_arcs = set()

    for arc_idx, arc in enumerate(arcs):
        if arc_idx in used_arcs:
            continue
            
        for i, line in enumerate(lines):
            if i in used_lines:
                continue

            door = classify_swing_door(arc, line)
            if door:
                swing_doors.append(door)
                used_lines.add(i)
                used_arcs.add(arc_idx)
                break  # One line per arc

    return swing_doors
