"""Geometry-based door classification rules."""
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
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


def _get_bbox_from_rect(rect) -> tuple:
    """Extract bbox from rect (handles both PyMuPDF Rect and tuple/list)."""
    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
        return rect
    elif hasattr(rect, 'x0'):
        return (rect.x0, rect.y0, rect.x1, rect.y1)
    return None


def _get_door_bbox_fast(door: Dict) -> tuple:
    """Fast bbox calculation - uses path_rect when available, avoids redundant calculations."""
    arc = door['arc']
    line = door['line']

    # Fast path: use path_rect if available (most common case)
    arc_rect = arc.get('path_rect')
    line_rect = line.get('path_rect')

    if arc_rect and line_rect:
        arc_bbox = _get_bbox_from_rect(arc_rect)
        line_bbox = _get_bbox_from_rect(line_rect)
        if arc_bbox and line_bbox:
            return (min(arc_bbox[0], line_bbox[0]), min(arc_bbox[1], line_bbox[1]),
                    max(arc_bbox[2], line_bbox[2]), max(arc_bbox[3], line_bbox[3]))

    # Fallback: calculate from points (rare case)
    if arc_rect:
        arc_bbox = _get_bbox_from_rect(arc_rect)
    else:
        cp = arc['control_points']
        # Use list comprehension with single pass
        xs = [p[0] for p in cp]
        ys = [p[1] for p in cp]
        arc_bbox = (min(xs), min(ys), max(xs), max(ys))

    if line_rect:
        line_bbox = _get_bbox_from_rect(line_rect)
    else:
        s, e = line['start'], line['end']
        line_bbox = (min(s[0], e[0]), min(s[1], e[1]),
                     max(s[0], e[0]), max(s[1], e[1]))

    return (min(arc_bbox[0], line_bbox[0]), min(arc_bbox[1], line_bbox[1]),
            max(arc_bbox[2], line_bbox[2]), max(arc_bbox[3], line_bbox[3]))


def check_arcs_touch(arc1: Dict, arc2: Dict, threshold: float) -> Tuple[bool, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Check if two arcs touch at one set of endpoints.

    Args:
        arc1: First arc dictionary with control_points
        arc2: Second arc dictionary with control_points
        threshold: Distance threshold for "touching"

    Returns:
        Tuple of (touches, endpoint_info) where:
        - touches: True if arcs touch at endpoints
        - endpoint_info: (arc1_touch_point, arc2_touch_point, arc1_other_point, arc2_other_point) if touching, None otherwise
    """
    arc1_start = np.array(arc1['control_points'][0])
    arc1_end = np.array(arc1['control_points'][-1])
    arc2_start = np.array(arc2['control_points'][0])
    arc2_end = np.array(arc2['control_points'][-1])

    # Check all 4 endpoint combinations (use squared distances for speed)
    threshold_sq = threshold * threshold
    distances_sq = [
        (np.sum((arc1_start - arc2_start) ** 2),
         arc1_start, arc2_start, arc1_end, arc2_end),
        (np.sum((arc1_start - arc2_end) ** 2),
         arc1_start, arc2_end, arc1_end, arc2_start),
        (np.sum((arc1_end - arc2_start) ** 2),
         arc1_end, arc2_start, arc1_start, arc2_end),
        (np.sum((arc1_end - arc2_end) ** 2),
         arc1_end, arc2_end, arc1_start, arc2_start)
    ]

    min_dist_sq, touch1, touch2, other1, other2 = min(
        distances_sq, key=lambda x: x[0])

    if min_dist_sq <= threshold_sq:
        return True, (touch1, touch2, other1, other2)
    return False, None


def classify_edge_double_doors(double_door_candidates: List[Dict], debug: bool = True) -> List[Dict]:
    """
    Classify edge double doors from double door candidate arcs.

    Edge double doors are two wide arcs (150-210°) that:
    1. Touch each other at one set of endpoints
    2. Have the other set of endpoints more than 1.5 radius away

    Args:
        double_door_candidates: List of arc dictionaries with 150-210° sweep angle
        debug: Enable debug output

    Returns:
        List of edge double door dictionaries with 'type' and 'bbox'
    """
    if len(double_door_candidates) < 2:
        return []

    edge_double_doors = []
    used_indices = set()

    if debug:
        print(f"\n{'='*60}")
        print(
            f"DEBUG classify_edge_double_doors: Checking {len(double_door_candidates)} double door candidates")

    # Pre-cache arc geometry to avoid repeated calculations in nested loops
    arc_cache = []
    for arc in double_door_candidates:
        if 'center' in arc and 'radius' in arc:
            center = np.array(arc['center'])
            radius = arc['radius']
        else:
            result = get_bezier_radius(arc['control_points'])
            if result is None:
                arc_cache.append(None)
                continue
            radius, center = result
        sweep = calculate_arc_sweep_angle(arc, radius)
        arc_cache.append((radius, center, sweep))

    for i in range(len(double_door_candidates)):
        if i in used_indices:
            continue

        arc1 = double_door_candidates[i]

        # Get arc1 geometry from cache
        if arc_cache[i] is None:
            if debug:
                print(f"  Arc {i}: Failed to calculate radius, skipping")
            continue
        radius1, center1, sweep1 = arc_cache[i]

        for j in range(i + 1, len(double_door_candidates)):
            if j in used_indices:
                continue

            arc2 = double_door_candidates[j]

            # Get arc2 geometry from cache
            if arc_cache[j] is None:
                if debug:
                    print(f"  Arc {j}: Failed to calculate radius, skipping")
                continue
            radius2, center2, sweep2 = arc_cache[j]

            if debug:
                print(f"\n  Checking arc pair ({i}, {j}):")
                print(
                    f"    Arc {i}: center=({int(center1[0])}, {int(center1[1])}), radius={radius1:.1f}, sweep={sweep1:.1f}°" if sweep1 else f"    Arc {i}: center=({int(center1[0])}, {int(center1[1])}), radius={radius1:.1f}, sweep=None")
                print(
                    f"    Arc {j}: center=({int(center2[0])}, {int(center2[1])}), radius={radius2:.1f}, sweep={sweep2:.1f}°" if sweep2 else f"    Arc {j}: center=({int(center2[0])}, {int(center2[1])}), radius={radius2:.1f}, sweep=None")

            # Use the average radius for distance checks
            avg_radius = (radius1 + radius2) / 2
            touch_threshold_combined = avg_radius * 0.75

            # Check if arcs touch at one set of endpoints
            touches, endpoint_info = check_arcs_touch(
                arc1, arc2, touch_threshold_combined)

            if not touches:
                # Calculate minimum distance between any endpoints for debug (use squared distances)
                arc1_start = np.array(arc1['control_points'][0])
                arc1_end = np.array(arc1['control_points'][-1])
                arc2_start = np.array(arc2['control_points'][0])
                arc2_end = np.array(arc2['control_points'][-1])

                min_dist_sq = min(
                    np.sum((arc1_start - arc2_start) ** 2),
                    np.sum((arc1_start - arc2_end) ** 2),
                    np.sum((arc1_end - arc2_start) ** 2),
                    np.sum((arc1_end - arc2_end) ** 2)
                )
                min_endpoint_dist = np.sqrt(min_dist_sq)

                if debug:
                    margin = touch_threshold_combined - min_endpoint_dist
                    print(f"    ✗ FAILED: Arcs do not touch")
                    print(
                        f"      Min endpoint distance: {min_endpoint_dist:.2f} (threshold: {touch_threshold_combined:.2f})")
                    print(
                        f"      Margin: {margin:.2f} (need to be {abs(margin):.2f} closer)")
                continue

            # Extract the other endpoints (the ones that don't touch)
            touch1, touch2, other1, other2 = endpoint_info
            touch_distance = np.linalg.norm(touch1 - touch2)

            # Check if the other endpoints are more than 1.5 radius away
            other_endpoints_distance = np.linalg.norm(other1 - other2)
            min_distance_required = 1.75 * avg_radius

            if debug:
                print(
                    f"    ✓ Touch check PASSED: distance={touch_distance:.2f} (threshold: {touch_threshold_combined:.2f})")
                print(
                    f"      Touch points: ({int(touch1[0])}, {int(touch1[1])}) <-> ({int(touch2[0])}, {int(touch2[1])})")
                print(
                    f"      Other endpoints: ({int(other1[0])}, {int(other1[1])}) <-> ({int(other2[0])}, {int(other2[1])})")
                print(
                    f"      Other endpoints distance: {other_endpoints_distance:.2f} (required: >{min_distance_required:.2f})")

            if other_endpoints_distance > min_distance_required:
                # This is an edge double door!
                # Calculate bounding box from both arcs
                arc1_rect = arc1.get('path_rect')
                arc2_rect = arc2.get('path_rect')

                if arc1_rect and arc2_rect:
                    arc1_bbox = _get_bbox_from_rect(arc1_rect)
                    arc2_bbox = _get_bbox_from_rect(arc2_rect)

                    if arc1_bbox and arc2_bbox:
                        combined_bbox = (
                            min(arc1_bbox[0], arc2_bbox[0]),
                            min(arc1_bbox[1], arc2_bbox[1]),
                            max(arc1_bbox[2], arc2_bbox[2]),
                            max(arc1_bbox[3], arc2_bbox[3])
                        )

                        edge_double_doors.append({
                            'type': 'double_door',
                            'bbox': combined_bbox
                        })
                        used_indices.add(i)
                        used_indices.add(j)

                        if debug:
                            margin = other_endpoints_distance - min_distance_required
                            print(f"    ✓✓ MATCHED: Edge double door found!")
                            print(
                                f"      Margin: {margin:.2f} above required distance")
                            print(
                                f"      Combined bbox: ({combined_bbox[0]:.1f}, {combined_bbox[1]:.1f}, {combined_bbox[2]:.1f}, {combined_bbox[3]:.1f})")
                        break  # One pair per arc
                elif debug:
                    print(f"    ✗ FAILED: Missing path_rect for one or both arcs")
            else:
                if debug:
                    margin = min_distance_required - other_endpoints_distance
                    print(f"    ✗ FAILED: Other endpoints too close")
                    print(
                        f"      Margin: {margin:.2f} (need to be {margin:.2f} farther apart)")

    if debug and edge_double_doors:
        print(f"\nDEBUG: Found {len(edge_double_doors)} edge double door(s)")

    return edge_double_doors


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

    threshold = arc_radius * .15  # if the arc touches LOOK HERE

    threshold_sq = threshold * threshold

    # Use squared distances to avoid sqrt (faster)
    distances_sq = [
        np.sum((arc_start - line_start) ** 2),
        np.sum((arc_start - line_end) ** 2),
        np.sum((arc_end - line_start) ** 2),
        np.sum((arc_end - line_end) ** 2)
    ]

    return min(distances_sq) <= threshold_sq


def classify_swing_door(arc: Dict, line: Dict, arc_radius: float, arc_center: np.ndarray, debug: bool = False, arc_idx: int = -1) -> Optional[Dict]:
    # Step 4: Angle Check (calculate first, needed for ratio scaling)
    sweep_angle = calculate_arc_sweep_angle(arc, arc_radius)
    # greater than 7.5 less than 120
    if sweep_angle is None or not (12.5 <= sweep_angle <= 103.5):  # LOOK HERE
        if debug:
            coord_str = f"center=({int(arc_center[0])}, {int(arc_center[1])})"
            print(
                f"  ✗ Rule 1 FAILED: Sweep angle {sweep_angle:.1f}° (required: 12.5-104°)" if sweep_angle else f"  ✗ Rule 1 FAILED: Sweep angle None")
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
    if sweep_angle >= 60:  # LOOK HERE ratios
        # Larger angles: tighter bounds (0.6x to 1.3x expected)
        min_ratio = expected_ratio * 0.45
        max_ratio = expected_ratio * 1.55
    elif sweep_angle >= 30:
        # Medium angles: moderate bounds (0.5x to 1.5x expected)
        min_ratio = expected_ratio * 0.6
        max_ratio = expected_ratio * 1.4
    else:
        # Small angles: wider bounds (0.4x to 2.0x expected)
        min_ratio = expected_ratio * 0.5
        max_ratio = expected_ratio * 1.5

    if not (min_ratio < ratio < max_ratio):
        if debug:
            print(
                f"  ✗ Rule 2 FAILED: Ratio {ratio:.2f} (line={line_length:.1f}, radius={arc_radius:.1f}, sweep={sweep_angle:.1f}°, expected={expected_ratio:.2f}, allowed: {min_ratio:.2f}-{max_ratio:.2f})")
        return None

    # Detailed debug output when passing (to see margins)
    if debug:
        # Rule 1 margins
        margin_from_min_angle = sweep_angle - 12.5
        margin_to_max_angle = 120 - sweep_angle
        closest_angle_limit = min(margin_from_min_angle, margin_to_max_angle)

        # Rule 2 margins
        margin_from_min_ratio = ratio - min_ratio
        margin_to_max_ratio = max_ratio - ratio
        closest_ratio_limit = min(margin_from_min_ratio, margin_to_max_ratio)

        print(
            f"  Rule 1 (Sweep Angle): {sweep_angle:.1f}° [required: 12.5-120°]")
        print(f"    Margin from min (12.5°): {margin_from_min_angle:.1f}°")
        print(f"    Margin to max (120°): {margin_to_max_angle:.1f}°")
        print(
            f"    Closest to limit: {closest_angle_limit:.1f}° {'(MIN)' if margin_from_min_angle < margin_to_max_angle else '(MAX)'}")

        print(
            f"  Rule 2 (Line/Radius Ratio): {ratio:.4f} [required: {min_ratio:.4f}-{max_ratio:.4f}]")
        print(f"    Line length: {line_length:.2f}, Radius: {arc_radius:.2f}")
        print(
            f"    Expected ratio: {expected_ratio:.4f} (for {sweep_angle:.1f}° sweep)")
        print(
            f"    Margin from min ({min_ratio:.4f}): {margin_from_min_ratio:.4f}")
        print(
            f"    Margin to max ({max_ratio:.4f}): {margin_to_max_ratio:.4f}")
        print(
            f"    Closest to limit: {closest_ratio_limit:.4f} {'(MIN)' if margin_from_min_ratio < margin_to_max_ratio else '(MAX)'}")

    return {
        "type": "swing_door",
        "arc": arc,
        "line": line,
        "arc_radius": arc_radius,
        "sweep_angle": sweep_angle,
        "center": arc_center
    }


def classify_swing_doors(arcs: List[Dict], lines: List[Dict], debug: bool = False, page_width: float = None, page_height: float = None, double_door_candidates: List[Dict] = None) -> Dict:
    """
    Classify all swing doors from arcs and lines.

    Args:
        arcs: List of arc dictionaries
        lines: List of line dictionaries
        debug: Enable debug output
        page_width: Page width for double door detection (optional)
        page_height: Page height for double door detection (optional)
        double_door_candidates: List of double door candidate arcs (150-210° sweep) for edge double door detection (optional)

    Returns:
        Dictionary with 'swing_doors' and 'double_doors' lists
    """
    swing_doors = []
    used_lines = set()
    used_arcs = set()

    for arc_idx, arc in enumerate(arcs):
        if arc_idx in used_arcs:
            continue

        # Cache arc geometry once per arc
        # For reconstructed arcs, use stored center if available (actual circle center)
        # Otherwise, use get_bezier_radius() which returns midpoint
        if 'center' in arc and 'radius' in arc:
            arc_center = np.array(arc['center'])
            arc_radius = arc['radius']
        else:
            result = get_bezier_radius(arc['control_points'])
            if result is None:
                continue  # Skip arcs that can't calculate radius
            arc_radius, arc_center = result

        if debug:
            coord_str = f"center=({int(arc_center[0])}, {int(arc_center[1])})"
            sweep = calculate_arc_sweep_angle(arc, arc_radius)
            print(f"\n{'='*60}")
            print(f"Arc {arc_idx} - {coord_str}")
            print(
                f"  Radius: {arc_radius:.1f}, Sweep: {sweep:.1f}°" if sweep else f"  Radius: {arc_radius:.1f}, Sweep: None")

            lines_checked = 0
            lines_passed_bbox = 0
            lines_passed_touch = 0
            closest_bbox_distance = float('inf')
            closest_touch_distance = float('inf')
            rejection_reason = None
            touch_info = None  # Store touch info for matching line

            # Pre-calculate arc bbox for efficiency using path_rect
        arc_rect = arc['path_rect']
        arc_x0, arc_y0, arc_x1, arc_y1 = arc_rect

        buffer = arc_radius * .15  # LOOK HERE bbox
        arc_bbox = (arc_x0 - buffer, arc_y0 - buffer,
                    arc_x1 + buffer, arc_y1 + buffer)

        # Cache arc control points as numpy arrays (avoid repeated conversions)
        arc_start_arr = np.array(arc['control_points'][0])
        arc_end_arr = np.array(arc['control_points'][-1])

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

            # Cache line points as numpy arrays (avoid repeated conversions)
            line_start_arr = np.array(line['start'])
            line_end_arr = np.array(line['end'])

            # Check touch using the actual function (for both logic and debug)
            touch_result = check_arc_line_touch(arc, line, arc_radius)
            touch_threshold = arc_radius * .15  # LOOK here for DEBUG

            # Calculate distances once (reused for debug and triangle angle test)
            # Use squared distances for finding minimum (faster), then sqrt only once
            dists_sq = [
                (np.sum((arc_start_arr - line_start_arr) ** 2),
                 arc_start_arr, line_start_arr, line_end_arr),
                (np.sum((arc_start_arr - line_end_arr) ** 2),
                 arc_start_arr, line_end_arr, line_start_arr),
                (np.sum((arc_end_arr - line_start_arr) ** 2),
                 arc_end_arr, line_start_arr, line_end_arr),
                (np.sum((arc_end_arr - line_end_arr) ** 2),
                 arc_end_arr, line_end_arr, line_start_arr)
            ]
            min_dist_sq, hinge_point, line_touch_point, line_other_point = min(
                dists_sq, key=lambda x: x[0])
            min_touch_dist = np.sqrt(min_dist_sq)

            if debug:
                if touch_result:
                    lines_passed_touch += 1
                    # Store touch info for the matching line (shown later)
                    touch_info = {
                        'min_distance': min_touch_dist,
                        'threshold': touch_threshold,
                        'margin': touch_threshold - min_touch_dist
                    }
                else:
                    if min_touch_dist < closest_touch_distance:
                        closest_touch_distance = min_touch_dist

            if not touch_result:
                continue

            # Filter: Triangle Angle Test - check if arc faces wrong way
            # For real doors: arc curves AWAY from line (sharp angle ~70-90°)
            # For false positives: arc curves TOWARD line (wide angle ~120-150°)

            # Point B: Midpoint of the arc (curve peak)
            # Use the midpoint of the arc's control points or calculate from the arc
            arc_midpoint = (arc_start_arr + arc_end_arr) / 2

            # Calculate angle at hinge (Point A)
            # Vector from hinge to curve peak
            vec_AB = arc_midpoint - hinge_point
            # Vector from hinge to line end
            vec_AC = line_other_point - hinge_point

            # Calculate angle using dot product
            norm_AB = np.linalg.norm(vec_AB)
            norm_AC = np.linalg.norm(vec_AC)

            if norm_AB > 1e-10 and norm_AC > 1e-10:
                cos_angle = np.clip(np.dot(vec_AB, vec_AC) /
                                    (norm_AB * norm_AC), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))

                # Reject if angle > 100° (arc curves toward line, not away)
                if angle_deg > 115:  # look here triangle test
                    if debug:
                        rejection_reason = f"Arc faces wrong way (hinge angle={angle_deg:.1f}° > 100°)"
                    continue
                elif debug and touch_result:  # Show triangle angle info when touch passes
                    margin_to_reject = 125 - angle_deg
                    print(f"  Triangle Angle Check: PASSED")
                    print(
                        f"    Hinge angle: {angle_deg:.1f}° (threshold: 100°)")
                    print(f"    Margin to rejection: {margin_to_reject:.1f}°")

            door = classify_swing_door(
                arc, line, arc_radius, arc_center, debug=debug, arc_idx=arc_idx)
            if door:
                if debug:
                    print(f"  ✓ MATCHED with line {i} - All rules passed!")
                    # Show touch info for the matching line
                    if touch_info is not None:
                        print(f"  Touch Check: PASSED")
                        print(
                            f"    Min distance: {touch_info['min_distance']:.2f} (threshold: {touch_info['threshold']:.2f})")
                        print(f"    Margin: {touch_info['margin']:.2f}")
                swing_doors.append(door)
                used_lines.add(i)
                used_arcs.add(arc_idx)
                rejection_reason = None  # Clear rejection reason on match
                break  # One line per arc
            elif debug and lines_passed_touch > 0:
                # Rule 1 or 2 failed (already printed in classify_swing_door)
                rejection_reason = "Rule 1 or 2 failed (see above)"

        if debug:
            if lines_passed_touch == 0:
                print(f"  ✗ NO MATCH - No lines passed touch check")
                print(
                    f"    Checked {lines_checked} lines, {lines_passed_bbox} passed bbox, {lines_passed_touch} passed touch")
                if closest_bbox_distance < float('inf'):
                    print(
                        f"    Closest bbox distance: {closest_bbox_distance:.1f} (need overlap)")
                if closest_touch_distance < float('inf'):
                    print(
                        f"    Closest touch distance: {closest_touch_distance:.1f} (threshold: {arc_radius * 0.03:.1f})")
                elif lines_passed_bbox == 0:
                    print(
                        f"    Closest bbox distance: {closest_bbox_distance:.1f} (no lines passed bbox)")
            elif rejection_reason:
                print(f"  ✗ NO MATCH - {rejection_reason}")
                print(
                    f"    Checked {lines_checked} lines, {lines_passed_bbox} passed bbox, {lines_passed_touch} passed touch")
            else:
                print(f"  ✓ MATCHED")
                print(
                    f"    Checked {lines_checked} lines, {lines_passed_bbox} passed bbox, {lines_passed_touch} passed touch")

    # Detect double doors: find overlapping swing doors (minimal overhead)
    double_doors = []
    if page_width is not None and page_height is not None and len(swing_doors) > 1:
        page_diagonal = np.sqrt(page_width**2 + page_height**2)
        buffer = page_diagonal * 0.000225  # overlapping single doors look here

        # Pre-calculate all bboxes once (fast path using path_rect)
        door_bboxes = []
        for door in swing_doors:
            bbox = _get_door_bbox_fast(door)
            door_bboxes.append(
                (bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer))

        # Simple O(n²) check - but only if there are few doors (early exit for many doors)
        # For typical cases (<100 doors), this is fast enough
        if len(swing_doors) > 100:
            # Skip double door detection if too many doors (performance)
            if debug:
                print(
                    f"\nDEBUG: Skipping double door detection ({len(swing_doors)} doors, too many)")
        else:
            used_indices = set()
            for i in range(len(swing_doors)):
                if i in used_indices:
                    continue

                bbox1 = door_bboxes[i]
                for j in range(i + 1, len(swing_doors)):
                    if j in used_indices:
                        continue

                    bbox2 = door_bboxes[j]

                    # Simple overlap check
                    if (bbox1[0] <= bbox2[2] and bbox2[0] <= bbox1[2] and
                            bbox1[1] <= bbox2[3] and bbox2[1] <= bbox1[3]):
                        # Overlap found - create double door
                        combined_bbox = (
                            min(bbox1[0], bbox2[0]),
                            min(bbox1[1], bbox2[1]),
                            max(bbox1[2], bbox2[2]),
                            max(bbox1[3], bbox2[3])
                        )

                        double_doors.append({
                            'type': 'double_door',
                            'bbox': combined_bbox
                        })
                        used_indices.add(i)
                        used_indices.add(j)
                        break  # One pair per door

            # Remove double doors from swing doors
            if used_indices:
                swing_doors = [door for idx, door in enumerate(
                    swing_doors) if idx not in used_indices]

            if debug and double_doors:
                print(f"\nDEBUG: Found {len(double_doors)} double door(s)")

    # Detect edge double doors from double door candidates (wide arcs that touch)
    if double_door_candidates is not None and len(double_door_candidates) >= 2:
        edge_double_doors = classify_edge_double_doors(
            double_door_candidates, debug=debug)
        # Merge edge double doors into the double_doors list
        double_doors.extend(edge_double_doors)
        if debug and edge_double_doors:
            print(
                f"DEBUG: Added {len(edge_double_doors)} edge double door(s) to double doors list")

    # Detect bifold doors from filtered lines (not already used)
    bifold_doors = classify_bifold_doors(
        lines, page_width=page_width, page_height=page_height, debug=debug)

    return {
        'swing_doors': swing_doors,
        'double_doors': double_doors,
        'bifold_doors': bifold_doors
    }


def classify_bifold_doors(lines: List[Dict], page_width: float, page_height: float, debug: bool = False) -> List[Dict]:
    """
    Detect bifold doors from V-shaped line segment pairs.

    A bifold door consists of two V-candidates (V-shapes) that meet certain
    geometric criteria including angle, length similarity, distance, and orientation.

    Args:
        lines: List of line dictionaries with 'start' and 'end' coordinates
        page_width: Width of the PDF page
        page_height: Height of the PDF page
        debug: Enable debug output

    Returns:
        List of bifold door dictionaries with 'type', 'v_candidates', 'line_ids', and 'bbox'
    """
    if len(lines) < 4:
        return []

    # Use gap tolerance based on page diagonal (consistent with geometry_analyzer)
    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    endpoint_tolerance = page_diagonal * 0.0009
    endpoint_tolerance_sq = endpoint_tolerance * endpoint_tolerance

    if debug:
        print(f"\nDEBUG classify_bifold_doors: Processing {len(lines)} lines")
        print(f"  Page diagonal: {page_diagonal:.1f}")
        print(
            f"  Endpoint tolerance (gap_tolerance): {endpoint_tolerance:.1f}")

    # Build endpoint -> line index mapping (spatial index for endpoints)
    # (x, y) rounded to tolerance -> list of (line_idx, endpoint_type)
    endpoint_map = {}

    def get_endpoint_key(point):
        """Round point to tolerance grid for endpoint matching.

        Using round() instead of truncation prevents nearby but distinct
        joints from being merged (critical for paired bifold doors).
        """
        return (int(round(point[0] / endpoint_tolerance)), int(round(point[1] / endpoint_tolerance)))

    for line_idx, line in enumerate(lines):
        start = np.array(line['start'])
        end = np.array(line['end'])

        start_key = get_endpoint_key(start)
        end_key = get_endpoint_key(end)

        if start_key not in endpoint_map:
            endpoint_map[start_key] = []
        if end_key not in endpoint_map:
            endpoint_map[end_key] = []

        endpoint_map[start_key].append((line_idx, 'start', start))
        endpoint_map[end_key].append((line_idx, 'end', end))

    if debug:
        endpoints_with_2_lines = sum(
            1 for conns in endpoint_map.values() if len(conns) == 2)
        endpoints_with_3plus_lines = sum(
            1 for conns in endpoint_map.values() if len(conns) >= 3)
        print(f"  Endpoints with exactly 2 lines: {endpoints_with_2_lines}")
        print(f"  Endpoints with 3+ lines: {endpoints_with_3plus_lines}")

    # Find V-candidates: endpoints where exactly 2 lines meet
    v_candidates = []

    for endpoint_key, connections in endpoint_map.items():
        if len(connections) != 2:  # Must have exactly 2 lines meeting
            continue

        (line1_idx, ep1_type, p1), (line2_idx, ep2_type, p2) = connections

        # Get actual joint point (average if not exactly same)
        joint_point = (p1 + p2) / 2

        # Get both lines
        line1 = lines[line1_idx]
        line2 = lines[line2_idx]

        # Get line vectors from joint point
        start1 = np.array(line1['start'])
        end1 = np.array(line1['end'])
        start2 = np.array(line2['start'])
        end2 = np.array(line2['end'])

        # Determine which endpoint of each line is at the joint
        # Vectors must point AWAY from the joint, not toward it
        # This prevents near-zero vectors which cause unstable angles & bisectors
        dist1_start = np.sum((start1 - joint_point) ** 2)
        dist1_end = np.sum((end1 - joint_point) ** 2)
        dist2_start = np.sum((start2 - joint_point) ** 2)
        dist2_end = np.sum((end2 - joint_point) ** 2)

        if dist1_start <= dist1_end:
            # Start is at joint, use end (pointing away)
            vec1 = end1 - joint_point
        else:
            # End is at joint, use start (pointing away)
            vec1 = start1 - joint_point

        if dist2_start <= dist2_end:
            # Start is at joint, use end (pointing away)
            vec2 = end2 - joint_point
        else:
            # End is at joint, use start (pointing away)
            vec2 = start2 - joint_point

        # Calculate line lengths
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)

        if len1 < 1e-5 or len2 < 1e-5:
            continue

        # Normalize vectors for angle calculation
        vec1_norm = vec1 / len1
        vec2_norm = vec2 / len2

        # Calculate angle between vectors (stricter: 25° to 70°)
        dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        if not (25 <= angle_deg <= 70):
            continue

        # Length similarity check (stricter: >= 0.65)
        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < 0.65:
            continue

        # Calculate bisector direction (average of normalized vectors)
        bisector = vec1_norm + vec2_norm
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm < 1e-5:
            continue
        bisector = bisector / bisector_norm

        # Store V-candidate
        avg_length = (len1 + len2) / 2
        v_candidates.append({
            'joint_point': joint_point,
            'avg_length': avg_length,
            'angle': angle_deg,
            'bisector': bisector,
            'line_indices': (line1_idx, line2_idx)
        })

    if debug:
        print(
            f"\nDEBUG classify_bifold_doors: Found {len(v_candidates)} V-candidates")
        if v_candidates:
            avg_lengths = [v['avg_length'] for v in v_candidates]
            angles = [v['angle'] for v in v_candidates]
            print(f"  Average lengths: {[f'{l:.1f}' for l in avg_lengths]}")
            print(f"  Angles: {[f'{a:.1f}°' for a in angles]}")

    if len(v_candidates) < 2:
        if debug:
            print(
                f"  Need at least 2 V-candidates to form bifold doors (found {len(v_candidates)})")
        return []

    # Build spatial index for V-candidates based on joint points
    # Use cell size based on average panel length
    avg_panel_length = np.mean([v['avg_length'] for v in v_candidates])
    cell_size = avg_panel_length * 0.3
    v_spatial_index = {}

    def get_v_cell_key(point):
        return (int(point[0] / cell_size), int(point[1] / cell_size))

    for v_idx, v_candidate in enumerate(v_candidates):
        joint_key = get_v_cell_key(v_candidate['joint_point'])
        if joint_key not in v_spatial_index:
            v_spatial_index[joint_key] = []
        v_spatial_index[joint_key].append(v_idx)

    # Pair V-candidates
    bifold_doors = []
    used_v_indices = set()

    for i, v1 in enumerate(v_candidates):
        if i in used_v_indices:
            continue

        joint1 = v1['joint_point']
        avg_len1 = v1['avg_length']
        angle1 = v1['angle']
        bisector1 = v1['bisector']

        # Search nearby V-candidates using spatial index
        search_radius = 1.7 * avg_len1
        search_cell_radius = math.ceil(search_radius / cell_size)
        joint_key1 = get_v_cell_key(joint1)

        candidates = set()
        for dx in range(-search_cell_radius, search_cell_radius + 1):
            for dy in range(-search_cell_radius, search_cell_radius + 1):
                neighbor_key = (joint_key1[0] + dx, joint_key1[1] + dy)
                if neighbor_key in v_spatial_index:
                    candidates.update(v_spatial_index[neighbor_key])

        # Filter candidates by actual distance and pairing criteria
        for j in candidates:
            if j <= i or j in used_v_indices:
                continue

            v2 = v_candidates[j]
            joint2 = v2['joint_point']
            avg_len2 = v2['avg_length']
            angle2 = v2['angle']
            bisector2 = v2['bisector']

            # Distance check: <= 1.7 * L (allows 15-unit spacing for 9-unit panels)
            avg_l = (avg_len1 + avg_len2) / 2
            joint_dist = np.sqrt(np.sum((joint1 - joint2) ** 2))
            if joint_dist > 1.7 * avg_l:
                if debug and i == 0:  # Debug first pair only
                    print(
                        f"  ✗ Distance rejected: {joint_dist:.1f} > {1.7*avg_l:.1f}")
                continue

            # PRIMARY: Bisector alignment (stricter: >= 0.88)
            bisector_dot = abs(np.dot(bisector1, bisector2))
            if bisector_dot < 0.88:
                if debug and i == 0:  # Debug first pair only
                    print(f"  ✗ Bisector rejected: {bisector_dot:.3f} < 0.88")
                continue

            # SECONDARY: Orientation match (stricter: <= 20°)
            angle_diff = abs(angle1 - angle2)
            angle_sum = abs(angle1 + angle2 - 180)
            if not (angle_diff <= 20 or angle_sum <= 20):
                if debug and i == 0:  # Debug first pair only
                    print(
                        f"  ✗ Angle rejected: diff={angle_diff:.1f}°, sum={angle_sum:.1f}°")
                continue

            # VALIDATION: Four free endpoints must align on a straight line
            # Collect endpoints from both V-candidates (excluding joint points)
            ep1 = np.array(lines[v1['line_indices'][0]]['start'])
            ep2 = np.array(lines[v1['line_indices'][0]]['end'])
            ep3 = np.array(lines[v1['line_indices'][1]]['start'])
            ep4 = np.array(lines[v1['line_indices'][1]]['end'])
            ep5 = np.array(lines[v2['line_indices'][0]]['start'])
            ep6 = np.array(lines[v2['line_indices'][0]]['end'])
            ep7 = np.array(lines[v2['line_indices'][1]]['start'])
            ep8 = np.array(lines[v2['line_indices'][1]]['end'])

            # Find 4 free endpoints (not at joints) - simple approach: filter by distance
            joint_tol_sq = (endpoint_tolerance * 3) ** 2
            all_eps = [ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8]
            free_endpoints = []
            for ep in all_eps:
                d1_sq = np.sum((ep - joint1) ** 2)
                d2_sq = np.sum((ep - joint2) ** 2)
                if d1_sq > joint_tol_sq and d2_sq > joint_tol_sq:
                    # Check for duplicates
                    is_new = True
                    for existing in free_endpoints:
                        if np.sum((ep - existing) ** 2) < joint_tol_sq:
                            is_new = False
                            break
                    if is_new:
                        free_endpoints.append(ep)

            # Need exactly 4 free endpoints for a bifold door
            if len(free_endpoints) != 4:
                continue

            # Fit line through first and last endpoint, check others
            p0, p1, p2, p3 = free_endpoints[0], free_endpoints[1], free_endpoints[2], free_endpoints[3]
            line_vec = p3 - p0
            line_len = np.linalg.norm(line_vec)

            if line_len < 1e-5:
                continue

            # Distance tolerance: 15% of average panel length
            tolerance = avg_l * 0.15
            tolerance_sq = tolerance * tolerance

            # Check distance of p1 and p2 from line (p0 and p3 define the line)
            all_aligned = True
            for pt in [p1, p2]:
                vec_to_pt = pt - p0
                cross = np.abs(np.cross(vec_to_pt, line_vec))
                dist_sq = (cross / line_len) ** 2
                if dist_sq > tolerance_sq:
                    all_aligned = False
                    if debug and i == 0:
                        print(
                            f"  ✗ Endpoint alignment rejected: dist={np.sqrt(dist_sq):.2f} > {tolerance:.2f}")
                    break

            if not all_aligned:
                continue  # Skip if alignment check failed

            # Valid pair found - create bifold door
            # Calculate bounding box from all 4 lines
            all_line_indices = set(v1['line_indices'] + v2['line_indices'])
            all_points = []
            for line_idx in all_line_indices:
                line = lines[line_idx]
                all_points.append(line['start'])
                all_points.append(line['end'])

            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            bifold_doors.append({
                'type': 'bifold_door',
                'v_candidates': [
                    {
                        'joint_point': tuple(v1['joint_point']),
                        'angle': v1['angle'],
                        'avg_length': v1['avg_length'],
                        'line_indices': v1['line_indices']
                    },
                    {
                        'joint_point': tuple(v2['joint_point']),
                        'angle': v2['angle'],
                        'avg_length': v2['avg_length'],
                        'line_indices': v2['line_indices']
                    }
                ],
                'line_ids': list(all_line_indices),
                'bbox': bbox
            })

            used_v_indices.add(i)
            used_v_indices.add(j)
            break  # One pair per V-candidate

    if debug:
        print(
            f"\nDEBUG classify_bifold_doors: Final result: {len(bifold_doors)} bifold door(s)")

    return bifold_doors
