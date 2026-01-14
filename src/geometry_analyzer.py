"""Wall detection and door candidate generation from geometry."""
import numpy as np
from typing import List, Dict, Tuple, Optional


def get_bezier_radius(control_points: List) -> Optional[Tuple[float, np.ndarray]]:
    """
    Calculates the radius and midpoint of the circular arc approximated by a cubic Bezier.
    Returns (radius, midpoint), or None if the points are collinear (a straight line).

    Uses the midpoint of the curve (t=0.5) and calculates the circumradius
    of the triangle formed by start, midpoint, and end points.

    Args:
        control_points: List of 4 control points for cubic Bezier

    Returns:
        Tuple of (radius, midpoint) as numpy array, or None if points are collinear
    """
    if len(control_points) != 4:
        return None  # PyMuPDF always uses cubic Bezier (4 points)

    p0 = np.array(control_points[0])  # Start
    p1 = np.array(control_points[1])  # Control 1
    p2 = np.array(control_points[2])  # Control 2
    p3 = np.array(control_points[3])  # End

    # 1. Calculate the point on the curve at t=0.5 (The geometric middle)
    # Formula: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    t = 0.5
    mid = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

    # 2. We now have 3 points on the arc: p0, mid, p3.
    # Use the geometric formula for circumradius of a triangle: R = (abc) / (4 * Area)

    # Side lengths of the triangle formed by these 3 points
    a = np.linalg.norm(mid - p0)
    b = np.linalg.norm(p3 - mid)
    c = np.linalg.norm(p3 - p0)

    # Area of the triangle using the "Shoelace" formula (cross product method)
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    x1, y1 = p0
    x2, y2 = mid
    x3, y3 = p3

    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    if area < 1e-5:
        return None  # Points are collinear, it's a straight line, not a curve.

    radius = (a * b * c) / (4 * area)

    return (radius, mid)


def separate_walls_and_doors(lines: List[Dict], arcs: List[Dict], door_min_percentile: float = 0, door_max_percentile: float = 100, thick_percentile: float = 70, min_length_percentile: float = 50) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Separate walls from door candidates in one pass.

    Doors are medium strokes (0-100th percentile), walls are thick lines.
    No length filtering - testing if stroke width is the issue.

    Args:
        lines: List of line dictionaries
        arcs: List of arc dictionaries
        door_min_percentile: Minimum percentile for door strokes (default 0)
        door_max_percentile: Maximum percentile for door strokes (default 100)
        thick_percentile: Percentile for thick strokes (walls)
        min_length_percentile: Not used (kept for compatibility)

    Returns:
        Tuple of (walls, door_lines, door_arcs)
    """
    if not lines and not arcs:
        return [], lines, arcs

    all_strokes = [l['stroke_width']
                   for l in lines] + [a['stroke_width'] for a in arcs]
    min_threshold = np.percentile(all_strokes, door_min_percentile)
    max_threshold = np.percentile(all_strokes, door_max_percentile)
    thick_threshold = np.percentile(all_strokes, thick_percentile)
    arc_min_threshold = np.percentile(all_strokes, 0)
    arc_max_threshold = np.percentile(all_strokes, 100)

    walls = []
    door_lines = []
    door_arcs = []

    # Separate lines into walls and door candidates - NO LENGTH FILTERING
    for line in lines:
        stroke_width = line['stroke_width']
        # Door candidates: all strokes (0-100th percentile)
        if min_threshold <= stroke_width <= max_threshold:
            door_lines.append(line)
        # Walls: thick strokes
        elif stroke_width >= thick_threshold:
            walls.append(line)

    # Separate arcs into door candidates - ALL ARCS (0-100th percentile)
    for arc in arcs:
        if arc_min_threshold <= arc['stroke_width'] <= arc_max_threshold:
            door_arcs.append(arc)

    return walls, door_lines, door_arcs


def analyze_geometry(lines: List[Dict], arcs: List[Dict], dashed_lines: List[Dict]) -> Dict:
    """
    Analyze geometry to find door candidates.

    Args:
        lines: List of line dictionaries
        arcs: List of arc dictionaries
        dashed_lines: List of dashed line dictionaries

    Returns:
        Dictionary with filtered lines, arcs, and door candidates
    """
    # Step 1: Separate walls from door candidates (combines wall detection and filtering)
    # Combine solid and dashed lines - door panels can be either
    all_lines = lines + dashed_lines
    walls, filtered_lines, filtered_arcs = separate_walls_and_doors(
        all_lines, arcs)

    print(
        f"DEBUG analyze_geometry: Number of filtered lines: {len(filtered_lines)}")
    print(f"DEBUG analyze_geometry: Number of walls: {len(walls)}")
    print(
        f"DEBUG analyze_geometry: Number of filtered arcs: {len(filtered_arcs)}")

    return {
        "filtered_lines": filtered_lines,
        "filtered_arcs": filtered_arcs,
        "door_candidate_arcs": filtered_arcs,  # Use filtered arcs directly
        "dashed_lines": dashed_lines,
        "walls": walls
    }
