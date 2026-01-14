"""Door candidate generation from geometry."""
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


def filter_door_candidates(lines: List[Dict], arcs: List[Dict], page_width: float, page_height: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter door candidates by stroke width.

    Filters lines and arcs to find potential door components based on stroke width.
    Removes dust (very short) and extremely long lines/arcs before calculating percentiles.

    Args:
        lines: List of line dictionaries
        arcs: List of arc dictionaries
        page_width: Width of the PDF page
        page_height: Height of the PDF page

    Returns:
        Tuple of (door_lines, door_arcs)
    """
    if not lines and not arcs:
        return lines, arcs

    # Calculate thresholds relative to page size
    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    MIN_LENGTH = page_diagonal * 0.0005  # 0.05% of page diagonal (dust)
    MAX_LENGTH = page_diagonal * 0.06  # 6% of page diagonal (extremely long)

    # Filter out dust and extremely long lines
    filtered_lines = []
    for line in lines:
        start = np.array(line['start'])
        end = np.array(line['end'])
        length = np.linalg.norm(end - start)
        if MIN_LENGTH <= length <= MAX_LENGTH:
            filtered_lines.append(line)

    # Filter out dust and extremely long arcs (using chord length)
    filtered_arcs_for_percentile = []
    for arc in arcs:
        control_points = arc['control_points']
        if len(control_points) == 4:
            p0 = np.array(control_points[0])
            p3 = np.array(control_points[3])
            chord_length = np.linalg.norm(p3 - p0)
            if MIN_LENGTH <= chord_length <= MAX_LENGTH:
                filtered_arcs_for_percentile.append(arc)

    # Hardcoded percentiles for door filtering
    door_min_percentile = 20
    door_max_percentile = 100

    # Calculate thresholds using filtered geometry only (no dust, no extremely long)
    line_strokes = [l['stroke_width'] for l in filtered_lines]
    arc_strokes = [a['stroke_width'] for a in filtered_arcs_for_percentile]
    all_strokes = line_strokes + arc_strokes

    if not all_strokes:
        return [], []

    min_threshold = np.percentile(all_strokes, door_min_percentile)
    max_threshold = np.percentile(all_strokes, door_max_percentile)

    # Calculate arc thresholds based on ARC stroke widths only, not combined
    if arc_strokes:
        arc_min_threshold = np.percentile(arc_strokes, 20)
        arc_max_threshold = np.percentile(arc_strokes, 90)
        print(
            f"DEBUG filter_door_candidates: Arc stroke width range: min={min(arc_strokes):.3f}, max={max(arc_strokes):.3f}, 40th={arc_min_threshold:.3f}, 80th={arc_max_threshold:.3f}")
    else:
        arc_min_threshold = 0
        arc_max_threshold = 0

    door_lines = []
    door_arcs = []

    # Filter lines by stroke width (using already length-filtered lines)
    for line in filtered_lines:
        stroke_width = line['stroke_width']
        if min_threshold <= stroke_width <= max_threshold:
            door_lines.append(line)

    # Filter arcs by stroke width (using already length-filtered arcs)
    arcs_filtered_out = 0
    for arc in filtered_arcs_for_percentile:
        if arc_min_threshold <= arc['stroke_width'] <= arc_max_threshold:
            door_arcs.append(arc)
        else:
            arcs_filtered_out += 1

    if arcs_filtered_out > 0:
        print(
            f"DEBUG filter_door_candidates: Filtered out {arcs_filtered_out} arcs, kept {len(door_arcs)} arcs")

    return door_lines, door_arcs


def analyze_geometry(lines: List[Dict], arcs: List[Dict], dashed_lines: List[Dict], page_width: float, page_height: float) -> Dict:
    """
    Analyze geometry to find door candidates.

    Args:
        lines: List of line dictionaries
        arcs: List of arc dictionaries
        dashed_lines: List of dashed line dictionaries
        page_width: Width of the PDF page
        page_height: Height of the PDF page

    Returns:
        Dictionary with filtered lines, arcs, and door candidates
    """
    # Combine solid and dashed lines - door panels can be either
    all_lines = lines + dashed_lines

    filtered_lines, filtered_arcs = filter_door_candidates(
        all_lines, arcs, page_width, page_height)

    print(
        f"DEBUG analyze_geometry: Number of filtered lines: {len(filtered_lines)}")
    print(
        f"DEBUG analyze_geometry: Number of filtered arcs: {len(filtered_arcs)}")

    return {
        "filtered_lines": filtered_lines,
        "filtered_arcs": filtered_arcs,
        "door_candidate_arcs": filtered_arcs,  # Use filtered arcs directly
        "dashed_lines": dashed_lines
    }
