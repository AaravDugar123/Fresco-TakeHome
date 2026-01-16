"""Geometry-based door classification rules."""
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

    # Check all 4 endpoint combinations
    distances = [
        (np.linalg.norm(arc1_start - arc2_start),
         arc1_start, arc2_start, arc1_end, arc2_end),
        (np.linalg.norm(arc1_start - arc2_end),
         arc1_start, arc2_end, arc1_end, arc2_start),
        (np.linalg.norm(arc1_end - arc2_start),
         arc1_end, arc2_start, arc1_start, arc2_end),
        (np.linalg.norm(arc1_end - arc2_end),
         arc1_end, arc2_end, arc1_start, arc2_start)
    ]

    min_dist, touch1, touch2, other1, other2 = min(
        distances, key=lambda x: x[0])

    if min_dist <= threshold:
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

    for i in range(len(double_door_candidates)):
        if i in used_indices:
            continue

        arc1 = double_door_candidates[i]

        # Get arc1 radius and geometry
        result1 = get_bezier_radius(arc1['control_points'])
        if result1 is None:
            if debug:
                print(f"  Arc {i}: Failed to calculate radius, skipping")
            continue
        radius1, center1 = result1

        # Get sweep angle for arc1
        sweep1 = calculate_arc_sweep_angle(arc1, radius1)

        for j in range(i + 1, len(double_door_candidates)):
            if j in used_indices:
                continue

            arc2 = double_door_candidates[j]

            # Get arc2 radius
            result2 = get_bezier_radius(arc2['control_points'])
            if result2 is None:
                if debug:
                    print(f"  Arc {j}: Failed to calculate radius, skipping")
                continue
            radius2, center2 = result2

            # Get sweep angle for arc2
            sweep2 = calculate_arc_sweep_angle(arc2, radius2)

            if debug:
                print(f"\n  Checking arc pair ({i}, {j}):")
                print(
                    f"    Arc {i}: center=({int(center1[0])}, {int(center1[1])}), radius={radius1:.1f}, sweep={sweep1:.1f}°" if sweep1 else f"    Arc {i}: center=({int(center1[0])}, {int(center1[1])}), radius={radius1:.1f}, sweep=None")
                print(
                    f"    Arc {j}: center=({int(center2[0])}, {int(center2[1])}), radius={radius2:.1f}, sweep={sweep2:.1f}°" if sweep2 else f"    Arc {j}: center=({int(center2[0])}, {int(center2[1])}), radius={radius2:.1f}, sweep=None")

            # Use the average radius for distance checks
            avg_radius = (radius1 + radius2) / 2
            touch_threshold_combined = avg_radius * 0.4

            # Check if arcs touch at one set of endpoints
            touches, endpoint_info = check_arcs_touch(
                arc1, arc2, touch_threshold_combined)

            if not touches:
                # Calculate minimum distance between any endpoints for debug
                arc1_start = np.array(arc1['control_points'][0])
                arc1_end = np.array(arc1['control_points'][-1])
                arc2_start = np.array(arc2['control_points'][0])
                arc2_end = np.array(arc2['control_points'][-1])

                min_endpoint_dist = min(
                    np.linalg.norm(arc1_start - arc2_start),
                    np.linalg.norm(arc1_start - arc2_end),
                    np.linalg.norm(arc1_end - arc2_start),
                    np.linalg.norm(arc1_end - arc2_end)
                )

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

    threshold = arc_radius * .02  # LOOK HERE
    # has to be greater than .05

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
    if sweep_angle >= 60:  # LOOK HERE
        # Larger angles: tighter bounds (0.6x to 1.3x expected)
        min_ratio = expected_ratio * 0.5
        max_ratio = expected_ratio * 1.3
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
            touch_threshold = arc_radius * 0.03

            if debug:
                # Always calculate touch distance for detailed info
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
            arc_start = np.array(arc['control_points'][0])
            arc_end = np.array(arc['control_points'][-1])
            line_start_arr = np.array(line['start'])
            line_end_arr = np.array(line['end'])

            # Find which arc endpoint touches the line (Point A - Hinge)
            dists = [
                (np.linalg.norm(arc_start - line_start_arr),
                 arc_start, line_start_arr, line_end_arr),
                (np.linalg.norm(arc_start - line_end_arr),
                 arc_start, line_end_arr, line_start_arr),
                (np.linalg.norm(arc_end - line_start_arr),
                 arc_end, line_start_arr, line_end_arr),
                (np.linalg.norm(arc_end - line_end_arr),
                 arc_end, line_end_arr, line_start_arr)
            ]
            _, hinge_point, line_touch_point, line_other_point = min(
                dists, key=lambda x: x[0])

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
                cos_angle = np.clip(np.dot(vec_AB, vec_AC) /
                                    (norm_AB * norm_AC), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))

                # Reject if angle > 100° (arc curves toward line, not away)
                if angle_deg > 100:
                    if debug:
                        rejection_reason = f"Arc faces wrong way (hinge angle={angle_deg:.1f}° > 100°)"
                    continue
                elif debug and touch_result:  # Show triangle angle info when touch passes
                    margin_to_reject = 100 - angle_deg
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
        buffer = page_diagonal * 0.001

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

    return {
        'swing_doors': swing_doors,
        'double_doors': double_doors
    }
