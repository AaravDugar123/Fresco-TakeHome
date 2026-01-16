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

    threshold = arc_radius * .03  # LOOK HERE
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
    if sweep_angle is None or not (12.5 <= sweep_angle <= 120):  # LOOK HERE
        if debug:
            coord_str = f"center=({int(arc_center[0])}, {int(arc_center[1])})"
            print(
                f"  ✗ Rule 1 FAILED: Sweep angle {sweep_angle:.1f}° (required: 12.5-120°)" if sweep_angle else f"  ✗ Rule 1 FAILED: Sweep angle None")
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

    return {
        "type": "swing_door",
        "arc": arc,
        "line": line,
        "arc_radius": arc_radius,
        "sweep_angle": sweep_angle,
        "center": arc_center
    }


def classify_swing_doors(arcs: List[Dict], lines: List[Dict], debug: bool = False, page_width: float = None, page_height: float = None) -> Dict:
    """
    Classify all swing doors from arcs and lines.

    Args:
        arcs: List of arc dictionaries
        lines: List of line dictionaries
        debug: Enable debug output
        page_width: Page width for double door detection (optional)
        page_height: Page height for double door detection (optional)

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

            door = classify_swing_door(
                arc, line, arc_radius, arc_center, debug=debug, arc_idx=arc_idx)
            if door:
                if debug:
                    print(f"  ✓ MATCHED with line {i} - All rules passed!")
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

    # Detect double doors: find overlapping swing doors
    double_doors = []
    if page_width is not None and page_height is not None and len(swing_doors) > 1:
        page_diagonal = np.sqrt(page_width**2 + page_height**2)
        buffer = page_diagonal * 0.001  # Small buffer for overlap detection
        
        # Calculate bbox for each door (combine arc and line)
        door_bboxes = []
        for door in swing_doors:
            arc = door['arc']
            line = door['line']
            
            # Get arc bbox
            arc_rect = arc.get('path_rect')
            if arc_rect:
                if isinstance(arc_rect, (list, tuple)) and len(arc_rect) >= 4:
                    arc_bbox = arc_rect
                else:
                    arc_bbox = (arc_rect.x0, arc_rect.y0, arc_rect.x1, arc_rect.y1)
            else:
                # Fallback: calculate from control points
                cp = arc['control_points']
                arc_bbox = (min(p[0] for p in cp), min(p[1] for p in cp),
                           max(p[0] for p in cp), max(p[1] for p in cp))
            
            # Get line bbox
            line_rect = line.get('path_rect')
            if line_rect:
                if isinstance(line_rect, (list, tuple)) and len(line_rect) >= 4:
                    line_bbox = line_rect
                else:
                    line_bbox = (line_rect.x0, line_rect.y0, line_rect.x1, line_rect.y1)
            else:
                # Fallback: calculate from start/end
                s, e = line['start'], line['end']
                line_bbox = (min(s[0], e[0]), min(s[1], e[1]), max(s[0], e[0]), max(s[1], e[1]))
            
            # Combined bbox with buffer
            min_x = min(arc_bbox[0], line_bbox[0]) - buffer
            min_y = min(arc_bbox[1], line_bbox[1]) - buffer
            max_x = max(arc_bbox[2], line_bbox[2]) + buffer
            max_y = max(arc_bbox[3], line_bbox[3]) + buffer
            
            door_bboxes.append((min_x, min_y, max_x, max_y))
        
        # Find overlapping pairs (efficient: only check each pair once)
        used_indices = set()
        for i in range(len(swing_doors)):
            if i in used_indices:
                continue
            
            bbox1 = door_bboxes[i]
            for j in range(i + 1, len(swing_doors)):
                if j in used_indices:
                    continue
                
                bbox2 = door_bboxes[j]
                
                # Check overlap
                if (bbox1[0] <= bbox2[2] and bbox2[0] <= bbox1[2] and
                    bbox1[1] <= bbox2[3] and bbox2[1] <= bbox1[3]):
                    # Found overlapping pair - create single double door with combined bbox
                    door1 = swing_doors[i]
                    door2 = swing_doors[j]
                    
                    # Calculate combined bbox
                    bbox1 = door_bboxes[i]
                    bbox2 = door_bboxes[j]
                    combined_bbox = (
                        min(bbox1[0], bbox2[0]),
                        min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]),
                        max(bbox1[3], bbox2[3])
                    )
                    
                    double_doors.append({
                        'type': 'double_door',
                        'arc': door1['arc'],  # Use first door's arc (or could combine)
                        'line': door1['line'],  # Use first door's line (or could combine)
                        'arc_radius': max(door1.get('arc_radius', 0), door2.get('arc_radius', 0)),
                        'sweep_angle': max(door1.get('sweep_angle', 0), door2.get('sweep_angle', 0)),
                        'center': door1.get('center'),  # Use first door's center
                        'bbox': combined_bbox,
                        'door1': door1,  # Keep references for debugging if needed
                        'door2': door2
                    })
                    used_indices.add(i)
                    used_indices.add(j)
                    break  # One pair per door
        
        # Remove double doors from swing doors list
        swing_doors = [door for idx, door in enumerate(swing_doors) if idx not in used_indices]
        
        if debug and double_doors:
            print(f"\nDEBUG: Found {len(double_doors)} double door(s)")
    
    return {
        'swing_doors': swing_doors,
        'double_doors': double_doors
    }
