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

    threshold = arc_radius * .01
    # has to be greater than .01

    distances = [
        np.linalg.norm(arc_start - line_start),
        np.linalg.norm(arc_start - line_end),
        np.linalg.norm(arc_end - line_start),
        np.linalg.norm(arc_end - line_end)
    ]

    return min(distances) <= threshold


def classify_swing_door(arc: Dict, line: Dict, arc_radius: float, arc_center: np.ndarray, debug: bool = False, arc_idx: int = -1) -> Optional[Dict]:
    # Step 4: Angle Check (calculate first, needed for ratio scaling)
    sweep_angle = calculate_arc_sweep_angle(arc, arc_radius)
    # greater than 7.5 less than 120
    if sweep_angle is None or not (7.5 <= sweep_angle <= 120):
        if debug:
            print(f"  DEBUG: Arc {arc_idx} - Rule 1 failed - Sweep angle: {sweep_angle:.1f}° (required: 7.5-120°)" if sweep_angle else f"  DEBUG: Arc {arc_idx} - Rule 1 failed - Sweep angle: None")
        return None

    # Step 2: Ratio Check (Length vs Radius) - scaled by sweep angle
    # For a circular arc: chord/radius = 2 * sin(sweep_angle / 2)
    # The door line length should be similar to the chord length
    line_start = np.array(line['start'])
    line_end = np.array(line['end'])
    line_length = np.linalg.norm(line_end - line_start)
    ratio = line_length / arc_radius

    # Calculate expected ratio for this sweep angle
    expected_ratio = 2 * np.sin(np.radians(sweep_angle / 2))

    # Scale bounds around expected ratio based on sweep angle
    # Wider tolerance for smaller angles (more variation in door designs)
    if sweep_angle >= 60:
        # Larger angles: tighter bounds (0.6x to 1.3x expected)
        min_ratio = expected_ratio * 0.6
        max_ratio = expected_ratio * 1.3
    elif sweep_angle >= 30:
        # Medium angles: moderate bounds (0.5x to 1.5x expected)
        min_ratio = expected_ratio * 0.55
        max_ratio = expected_ratio * 1.45
    else:
        # Small angles: wider bounds (0.4x to 2.0x expected)
        min_ratio = expected_ratio * 0.45
        max_ratio = expected_ratio * 1.55

    if not (min_ratio < ratio < max_ratio):
        if debug:
            print(f"  DEBUG: Arc {arc_idx} - Rule 2 failed - Ratio: {ratio:.2f} (line_length={line_length:.2f}, arc_radius={arc_radius:.2f}, sweep={sweep_angle:.1f}°, expected={expected_ratio:.2f}, allowed: {min_ratio:.2f}-{max_ratio:.2f})")
        return None

    return {
        "type": "swing_door",
        "arc": arc,
        "line": line,
        "arc_radius": arc_radius,
        "sweep_angle": sweep_angle,
        "center": arc_center
    }


def classify_swing_doors(arcs: List[Dict], lines: List[Dict], debug: bool = False) -> List[Dict]:
    """
    Classify all swing doors from arcs and lines.

    Args:
        arcs: List of arc dictionaries
        lines: List of line dictionaries
        debug: Enable debug output

    Returns:
        List of classified swing door dictionaries
    """
    swing_doors = []
    used_lines = set()
    used_arcs = set()

    for arc_idx, arc in enumerate(arcs):
        if arc_idx in used_arcs:
            continue

        # Cache arc geometry once per arc
        result = get_bezier_radius(arc['control_points'])
        if result is None:
            continue  # Skip arcs that can't calculate radius
        arc_radius, arc_center = result

        if debug:
            print(f"\nDEBUG: Testing arc {arc_idx}")
            sweep = calculate_arc_sweep_angle(arc, arc_radius)
            print(
                f"  Arc radius: {arc_radius:.2f}, sweep: {sweep:.1f}°" if sweep else f"  Arc radius: {arc_radius:.2f}, sweep: None")

            lines_checked = 0
            lines_passed_bbox = 0
            lines_passed_touch = 0
            closest_bbox_distance = float('inf')
            closest_touch_distance = float('inf')

            # Pre-calculate arc bbox for efficiency using path_rect
        arc_rect = arc['path_rect']
        arc_x0, arc_y0, arc_x1, arc_y1 = arc_rect

        buffer = arc_radius * .2
        arc_bbox = (arc_x0 - buffer, arc_y0 - buffer,
                    arc_x1 + buffer, arc_y1 + buffer)

        for i, line in enumerate(lines):
            if i in used_lines:
                continue

            if debug:
                lines_checked += 1  # Count ALL lines evaluated

            # Bounding Box Overlap Check (Fast spatial filter)
            line_rect = line['path_rect']
            line_x0, line_y0, line_x1, line_y1 = line_rect
            line_bbox = (line_x0, line_y0, line_x1, line_y1)

            # Calculate bbox distance for debug (even if it fails)
            if debug:
                if (arc_bbox[0] <= line_bbox[2] and line_bbox[0] <= arc_bbox[2] and
                        arc_bbox[1] <= line_bbox[3] and line_bbox[1] <= arc_bbox[3]):
                    bbox_distance = 0  # Overlapping
                    lines_passed_bbox += 1
                else:
                    # Calculate minimum distance between bboxes
                    dx = max(arc_bbox[0] - line_bbox[2],
                             line_bbox[0] - arc_bbox[2], 0)
                    dy = max(arc_bbox[1] - line_bbox[3],
                             line_bbox[1] - arc_bbox[3], 0)
                    bbox_distance = (dx*dx + dy*dy) ** 0.5

                if bbox_distance < closest_bbox_distance:
                    closest_bbox_distance = bbox_distance

            if not (arc_bbox[0] <= line_bbox[2] and line_bbox[0] <= arc_bbox[2] and
                    arc_bbox[1] <= line_bbox[3] and line_bbox[1] <= arc_bbox[3]):
                continue

            # Check touch using the actual function (for both logic and debug)
            touch_result = check_arc_line_touch(arc, line, arc_radius)

            if debug:
                if touch_result:
                    lines_passed_touch += 1
                else:
                    # Calculate touch distances for debug stats (only when touch fails)
                    arc_start = np.array(arc['control_points'][0])
                    arc_end = np.array(arc['control_points'][-1])
                    line_start_arr = np.array(line['start'])
                    line_end_arr = np.array(line['end'])

                    distances = [
                        np.linalg.norm(arc_start - line_start_arr),
                        np.linalg.norm(arc_start - line_end_arr),
                        np.linalg.norm(arc_end - line_start_arr),
                        np.linalg.norm(arc_end - line_end_arr)
                    ]
                    min_touch_dist = min(distances)

                    if min_touch_dist < closest_touch_distance:
                        closest_touch_distance = min_touch_dist

            if not touch_result:
                continue

            # Filter: Triangle Angle Test - check if arc faces wrong way
            # For real doors: arc curves AWAY from line (sharp angle ~70-90°)
            # For false positives: arc curves TOWARD line (wide angle ~120-150°)
            arc_start = np.array(arc['control_points'][0])
            arc_end = np.array(arc['control_points'][-1])
            line_start_arr = np.array(line['start'])
            line_end_arr = np.array(line['end'])
            
            # Find which arc endpoint touches the line (Point A - Hinge)
            dists = [
                (np.linalg.norm(arc_start - line_start_arr), arc_start, line_start_arr, line_end_arr),
                (np.linalg.norm(arc_start - line_end_arr), arc_start, line_end_arr, line_start_arr),
                (np.linalg.norm(arc_end - line_start_arr), arc_end, line_start_arr, line_end_arr),
                (np.linalg.norm(arc_end - line_end_arr), arc_end, line_end_arr, line_start_arr)
            ]
            _, hinge_point, line_touch_point, line_other_point = min(dists, key=lambda x: x[0])
            
            # Point B: Midpoint of the arc (curve peak)
            # Use the midpoint of the arc's control points or calculate from the arc
            arc_midpoint = (arc_start + arc_end) / 2
            
            # Calculate angle at hinge (Point A)
            # Vector from hinge to curve peak
            vec_AB = arc_midpoint - hinge_point
            # Vector from hinge to line end
            vec_AC = line_other_point - hinge_point
            
            # Calculate angle using dot product
            norm_AB = np.linalg.norm(vec_AB)
            norm_AC = np.linalg.norm(vec_AC)
            
            if norm_AB > 1e-10 and norm_AC > 1e-10:
                cos_angle = np.clip(np.dot(vec_AB, vec_AC) / (norm_AB * norm_AC), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                
                # Reject if angle > 100° (arc curves toward line, not away)
                if angle_deg > 100:
                    if debug:
                        print(f"  DEBUG: Arc {arc_idx} - Rejected: arc faces wrong way (hinge angle={angle_deg:.1f}° > 100°)")
                    continue

            door = classify_swing_door(
                arc, line, arc_radius, arc_center, debug=debug, arc_idx=arc_idx)
            if door:
                if debug:
                    print(
                        f"  DEBUG: Arc {arc_idx} matched with line {i} - All rules passed!")
                swing_doors.append(door)
                used_lines.add(i)
                used_arcs.add(arc_idx)
                break  # One line per arc

        if debug:
            print(
                f"  Checked {lines_checked} lines, {lines_passed_bbox} passed bbox, {lines_passed_touch} passed touch")
            if lines_passed_touch == 0:
                print(
                    f"  DEBUG: Arc {arc_idx} - No lines passed touch check (all failed before Rule 1/2)")
                if closest_bbox_distance < float('inf'):
                    print(
                        f"  Closest bbox distance: {closest_bbox_distance:.2f} (threshold: overlap required)")
                if closest_touch_distance < float('inf'):
                    print(
                        f"  Closest touch distance: {closest_touch_distance:.2f} (threshold: {arc_radius * 0.8:.2f})")
                elif lines_passed_bbox == 0:
                    print(
                        f"  Closest bbox distance: {closest_bbox_distance:.2f} (no lines passed bbox check)")

    return swing_doors
