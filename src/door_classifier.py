"""Geometry-based door classification rules."""
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.geometry_analyzer import get_bezier_radius


# ============================================================================
# UNIVERSAL FUNCS FOR DOOR CLASSIFICATION
# ============================================================================

def calculate_arc_sweep_angle(arc: Dict, radius: float) -> Optional[float]:
    """Calculate the sweep angleusing chord/radius ratio"""
    pts = arc['control_points']
    if len(pts) != 4 or radius <= 0:
        return None

    p0, p3 = np.array(pts[0]), np.array(pts[3])
    chord_len = np.linalg.norm(p3 - p0)

    # Cap chord to diameter
    if chord_len > 2 * radius:
        chord_len = 2 * radius

    angle_rad = 2 * np.arcsin(chord_len / (2 * radius))
    return np.degrees(angle_rad)


def _get_bbox_from_rect(rect) -> tuple:
    """Extract bbox from rect using pyMuPDF"""
    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
        return rect
    elif hasattr(rect, 'x0'):
        return (rect.x0, rect.y0, rect.x1, rect.y1)
    return None


def _get_door_bbox_fast(door: Dict) -> tuple:
    """Fast bbox calculation using path_rect"""
    arc, line = door['arc'], door['line']

    arc_rect = arc.get('path_rect')
    line_rect = line.get('path_rect')

    if arc_rect and line_rect:
        arc_bbox = _get_bbox_from_rect(arc_rect)
        line_bbox = _get_bbox_from_rect(line_rect)
        if arc_bbox and line_bbox:
            return (min(arc_bbox[0], line_bbox[0]), min(arc_bbox[1], line_bbox[1]),
                    max(arc_bbox[2], line_bbox[2]), max(arc_bbox[3], line_bbox[3]))

    # Fallback: calculate from points
    if arc_rect:
        arc_bbox = _get_bbox_from_rect(arc_rect)
    else:
        cp = arc['control_points']
        xs, ys = [p[0] for p in cp], [p[1] for p in cp]
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
    """Check if two arcs touch at endpoints Return (touches, endpoint_info)"""
    arc1_start = np.array(arc1['control_points'][0])
    arc1_end = np.array(arc1['control_points'][-1])
    arc2_start = np.array(arc2['control_points'][0])
    arc2_end = np.array(arc2['control_points'][-1])

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


def check_arc_line_touch(arc: Dict, line: Dict, arc_radius: float) -> bool:
    """Check if arc and line touch at endpoints"""
    arc_start = np.array(arc['control_points'][0])
    arc_end = np.array(arc['control_points'][-1])
    line_start = np.array(line['start'])
    line_end = np.array(line['end'])

    threshold_sq = (arc_radius * 0.15) ** 2
    distances_sq = [
        np.sum((arc_start - line_start) ** 2),
        np.sum((arc_start - line_end) ** 2),
        np.sum((arc_end - line_start) ** 2),
        np.sum((arc_end - line_end) ** 2)
    ]

    return min(distances_sq) <= threshold_sq


# ============================================================================
# SWING DOOR CLASSIFICATION
# ============================================================================

def classify_swing_door(arc: Dict, line: Dict, arc_radius: float, arc_center: np.ndarray, debug: bool = False, arc_idx: int = -1) -> Optional[Dict]:
    """Classify a single swing door with arcs and pairs"""
    sweep_angle = calculate_arc_sweep_angle(arc, arc_radius)
    if sweep_angle is None or not (12.5 <= sweep_angle <= 103.5):
        return None

    # Ratio check: line length vs radius (scaled by sweep angle)
    line_start = np.array(line['start'])
    line_end = np.array(line['end'])
    line_length = np.linalg.norm(line_end - line_start)
    ratio = line_length / arc_radius
    expected_ratio = 2 * np.sin(np.radians(sweep_angle / 2))

    # Scale bounds based on sweep angle
    if sweep_angle >= 60:
        min_ratio, max_ratio = expected_ratio * 0.45, expected_ratio * 1.55
    elif sweep_angle >= 30:
        min_ratio, max_ratio = expected_ratio * 0.6, expected_ratio * 1.4
    else:
        min_ratio, max_ratio = expected_ratio * 0.5, expected_ratio * 1.5

    if not (min_ratio < ratio < max_ratio):
        return None

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
    Classify swing doors from arcs and lines.
    Returns dict with 'swing_doors', 'double_doors', and 'bifold_doors' lists.
    """
    swing_doors = []
    used_lines = set()
    used_arcs = set()

    for arc_idx, arc in enumerate(arcs):
        if arc_idx in used_arcs:
            continue

        # Get arc geometry
        if 'center' in arc and 'radius' in arc:
            arc_center = np.array(arc['center'])
            arc_radius = arc['radius']
        else:
            result = get_bezier_radius(arc['control_points'])
            if result is None:
                continue
            arc_radius, arc_center = result

        arc_rect = arc['path_rect']
        buffer = arc_radius * 0.15
        arc_bbox = (arc_rect[0] - buffer, arc_rect[1] - buffer,
                    arc_rect[2] + buffer, arc_rect[3] + buffer)

        arc_start_arr = np.array(arc['control_points'][0])
        arc_end_arr = np.array(arc['control_points'][-1])

        for i, line in enumerate(lines):
            if i in used_lines:
                continue

            # Bbox overlap check
            line_rect = line['path_rect']
            line_bbox = (line_rect[0], line_rect[1],
                         line_rect[2], line_rect[3])
            if not (arc_bbox[0] <= line_bbox[2] and line_bbox[0] <= arc_bbox[2] and
                    arc_bbox[1] <= line_bbox[3] and line_bbox[1] <= arc_bbox[3]):
                continue

            # Touch check
            if not check_arc_line_touch(arc, line, arc_radius):
                continue

            # Triangle angle test (reject if arc curves toward line)
            line_start_arr = np.array(line['start'])
            line_end_arr = np.array(line['end'])

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

            arc_midpoint = (arc_start_arr + arc_end_arr) / 2
            vec_AB = arc_midpoint - hinge_point
            vec_AC = line_other_point - hinge_point

            norm_AB = np.linalg.norm(vec_AB)
            norm_AC = np.linalg.norm(vec_AC)

            if norm_AB > 1e-10 and norm_AC > 1e-10:
                cos_angle = np.clip(np.dot(vec_AB, vec_AC) /
                                    (norm_AB * norm_AC), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                if angle_deg > 115:  #cureves away
                    continue

            # Classify swing door
            door = classify_swing_door(
                arc, line, arc_radius, arc_center, debug=debug, arc_idx=arc_idx)
            if door:
                swing_doors.append(door)
                used_lines.add(i)
                used_arcs.add(arc_idx)
                break

    # Detect double doors from overlapping swing doors
    double_doors = []
    if page_width is not None and page_height is not None and len(swing_doors) > 1 and len(swing_doors) <= 100:
        page_diagonal = np.sqrt(page_width**2 + page_height**2)
        buffer = page_diagonal * 0.000225

        door_bboxes = []
        for door in swing_doors:
            bbox = _get_door_bbox_fast(door)
            door_bboxes.append(
                (bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer))

        used_indices = set()
        for i in range(len(swing_doors)):
            if i in used_indices:
                continue
            bbox1 = door_bboxes[i]
            for j in range(i + 1, len(swing_doors)):
                if j in used_indices:
                    continue
                bbox2 = door_bboxes[j]

                if (bbox1[0] <= bbox2[2] and bbox2[0] <= bbox1[2] and
                        bbox1[1] <= bbox2[3] and bbox2[1] <= bbox1[3]):
                    combined_bbox = (
                        min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])
                    )
                    double_doors.append(
                        {'type': 'double_door', 'bbox': combined_bbox})
                    used_indices.add(i)
                    used_indices.add(j)
                    break

        if used_indices:
            swing_doors = [door for idx, door in enumerate(
                swing_doors) if idx not in used_indices]

        if debug:
            for door in double_doors:
                bbox = door.get('bbox', (0, 0, 0, 0))
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                print(
                    f"Double door: center=({int(center[0])}, {int(center[1])})")
            if double_doors:
                print(f"Found {len(double_doors)} double door(s)")

    # Detect wide double doors from double door candidates
    if double_door_candidates is not None and len(double_door_candidates) >= 2:
        edge_double_doors = classify_edge_double_doors(
            double_door_candidates, debug=False)
        double_doors.extend(edge_double_doors)

    # Detect bifold doors
    bifold_doors = classify_bifold_doors(
        lines, page_width=page_width, page_height=page_height, debug=debug)

    if debug:
        for door in swing_doors:
            center = door.get('center', np.array([0, 0]))
            print(f"Swing door: center=({int(center[0])}, {int(center[1])})")
        if swing_doors:
            print(f"Found {len(swing_doors)} swing door(s)")

    return {
        'swing_doors': swing_doors,
        'double_doors': double_doors,
        'bifold_doors': bifold_doors
    }


# ============================================================================
# DOUBLE DOOR CLASSIFICATION
# ============================================================================

def classify_edge_double_doors(double_door_candidates: List[Dict], debug: bool = True) -> List[Dict]:
    """
    Classify edge double doors from double door candidate arcs
    two wide arcs (150-210°) that:
    1. Touch at one set of endpoints (within 0.75 * radius)
    2. Have other endpoints > 1.75 * radius apart
    """
    if len(double_door_candidates) < 2:
        return []

    arc_cache = []
    arc_cache = []
    for arc in double_door_candidates:
        if 'center' in arc and 'radius' in arc:
            center, radius = np.array(arc['center']), arc['radius']
        else:
            result = get_bezier_radius(arc['control_points'])
            if result is None:
                arc_cache.append(None)
                continue
            radius, center = result
        sweep = calculate_arc_sweep_angle(arc, radius)
        arc_cache.append((radius, center, sweep))

    edge_double_doors = []
    used_indices = set()

    for i in range(len(double_door_candidates)):
        if i in used_indices or arc_cache[i] is None:
            continue

        arc1 = double_door_candidates[i]
        radius1, center1, sweep1 = arc_cache[i]

        for j in range(i + 1, len(double_door_candidates)):
            if j in used_indices or arc_cache[j] is None:
                continue

            arc2 = double_door_candidates[j]
            radius2, center2, sweep2 = arc_cache[j]

            avg_radius = (radius1 + radius2) / 2
            touch_threshold = avg_radius * 0.75

            # Check if arcs touch
            touches, endpoint_info = check_arcs_touch(
                arc1, arc2, touch_threshold)
            if not touches:
                continue

            # Check other endpoints are far enough apart
            touch1, touch2, other1, other2 = endpoint_info
            other_distance = np.linalg.norm(other1 - other2)
            min_distance = 1.75 * avg_radius

            if other_distance > min_distance:
                # Calculate bounding box
                arc1_rect = arc1.get('path_rect')
                arc2_rect = arc2.get('path_rect')

                if arc1_rect and arc2_rect:
                    arc1_bbox = _get_bbox_from_rect(arc1_rect)
                    arc2_bbox = _get_bbox_from_rect(arc2_rect)

                    if arc1_bbox and arc2_bbox:
                        combined_bbox = (
                            min(arc1_bbox[0], arc2_bbox[0]), min(
                                arc1_bbox[1], arc2_bbox[1]),
                            max(arc1_bbox[2], arc2_bbox[2]), max(
                                arc1_bbox[3], arc2_bbox[3])
                        )
                        edge_double_doors.append(
                            {'type': 'double_door', 'bbox': combined_bbox})
                        used_indices.add(i)
                        used_indices.add(j)
                        break

    return edge_double_doors


# ============================================================================
# BIFOLD DOOR CLASSIFICATION
# ============================================================================

def classify_bifold_doors(lines: List[Dict], page_width: float, page_height: float, debug: bool = False) -> List[Dict]:
    """
    Detect bifold doors from V-shaped line segment pairs.

    A bifold door consists of two V-candidates that meet:
    - Distance between joints <= 1.7 * panel_length
    - parraele alignment >= 0.88
    - Angle match <= 20°
    - you can make a line out of it
    """
    if len(lines) < 4:
        return []

    # Use gap tolerance for endpoint matching
    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    endpoint_tolerance = page_diagonal * 0.0009
    endpoint_tolerance_sq = endpoint_tolerance * endpoint_tolerance

    endpoint_map = {}
    endpoint_map = {}

    def get_endpoint_key(point):
        return (int(round(point[0] / endpoint_tolerance)), int(round(point[1] / endpoint_tolerance)))

    for line_idx, line in enumerate(lines):
        start, end = np.array(line['start']), np.array(line['end'])
        start_key, end_key = get_endpoint_key(start), get_endpoint_key(end)

        if start_key not in endpoint_map:
            endpoint_map[start_key] = []
        if end_key not in endpoint_map:
            endpoint_map[end_key] = []

        endpoint_map[start_key].append((line_idx, 'start', start))
        endpoint_map[end_key].append((line_idx, 'end', end))

    # Find V-candidates
    v_candidates = []

    for endpoint_key, connections in endpoint_map.items():
        if len(connections) != 2:
            continue

        (line1_idx, _, p1), (line2_idx, _, p2) = connections
        joint_point = (p1 + p2) / 2

        line1, line2 = lines[line1_idx], lines[line2_idx]
        start1, end1 = np.array(line1['start']), np.array(line1['end'])
        start2, end2 = np.array(line2['start']), np.array(line2['end'])

        # Get vectors 
        dist1_start = np.sum((start1 - joint_point) ** 2)
        dist1_end = np.sum((end1 - joint_point) ** 2)
        dist2_start = np.sum((start2 - joint_point) ** 2)
        dist2_end = np.sum((end2 - joint_point) ** 2)

        vec1 = (
            end1 - joint_point) if dist1_start <= dist1_end else (start1 - joint_point)
        vec2 = (
            end2 - joint_point) if dist2_start <= dist2_end else (start2 - joint_point)

        len1, len2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if len1 < 1e-5 or len2 < 1e-5:
            continue

        vec1_norm, vec2_norm = vec1 / len1, vec2 / len2

        # angle check
        dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_product))
        if not (25 <= angle_deg <= 70):
            continue

        # Check same length
        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < 0.65:
            continue

        # bisector or parrelness
        bisector = vec1_norm + vec2_norm
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm < 1e-5:
            continue
        bisector = bisector / bisector_norm

        v_candidates.append({
            'joint_point': joint_point,
            'avg_length': (len1 + len2) / 2,
            'angle': angle_deg,
            'bisector': bisector,
            'line_indices': (line1_idx, line2_idx)
        })

    if len(v_candidates) < 2:
        return []

    # spatial graph for speed!
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

        # Search nearby candidates
        search_radius = 1.7 * avg_len1
        search_cell_radius = math.ceil(search_radius / cell_size)
        joint_key1 = get_v_cell_key(joint1)

        candidates = set()
        for dx in range(-search_cell_radius, search_cell_radius + 1):
            for dy in range(-search_cell_radius, search_cell_radius + 1):
                neighbor_key = (joint_key1[0] + dx, joint_key1[1] + dy)
                if neighbor_key in v_spatial_index:
                    candidates.update(v_spatial_index[neighbor_key])

        for j in candidates:
            if j <= i or j in used_v_indices:
                continue

            v2 = v_candidates[j]
            joint2 = v2['joint_point']
            avg_len2 = v2['avg_length']
            angle2 = v2['angle']
            bisector2 = v2['bisector']

            # Distance check: <= 1.7 * L
            avg_l = (avg_len1 + avg_len2) / 2
            joint_dist = np.sqrt(np.sum((joint1 - joint2) ** 2))
            if joint_dist > 1.7 * avg_l:
                continue

            # Bisector alignment: >= 0.88
            bisector_dot = abs(np.dot(bisector1, bisector2))
            if bisector_dot < 0.88:
                continue

            # Angle match: <= 20° or mirrored
            angle_diff = abs(angle1 - angle2)
            angle_sum = abs(angle1 + angle2 - 180)
            if not (angle_diff <= 20 or angle_sum <= 20):
                continue

            # Endpoint alignment check: four free endpoints must lie on a line
            all_eps = [
                np.array(lines[v1['line_indices'][0]]['start']),
                np.array(lines[v1['line_indices'][0]]['end']),
                np.array(lines[v1['line_indices'][1]]['start']),
                np.array(lines[v1['line_indices'][1]]['end']),
                np.array(lines[v2['line_indices'][0]]['start']),
                np.array(lines[v2['line_indices'][0]]['end']),
                np.array(lines[v2['line_indices'][1]]['start']),
                np.array(lines[v2['line_indices'][1]]['end'])
            ]

            # Filter free endpoints 
            joint_tol_sq = (endpoint_tolerance * 3) ** 2
            free_endpoints = []
            for ep in all_eps:
                d1_sq = np.sum((ep - joint1) ** 2)
                d2_sq = np.sum((ep - joint2) ** 2)
                if d1_sq > joint_tol_sq and d2_sq > joint_tol_sq:
                    is_new = True
                    for existing in free_endpoints:
                        if np.sum((ep - existing) ** 2) < joint_tol_sq:
                            is_new = False
                            break
                    if is_new:
                        free_endpoints.append(ep)

            if len(free_endpoints) != 4:
                continue

            # Check all 4 endpoints align on a line
            p0, p1, p2, p3 = free_endpoints[0], free_endpoints[1], free_endpoints[2], free_endpoints[3]
            line_vec = p3 - p0
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-5:
                continue

            tolerance_sq = (avg_l * 0.15) ** 2
            all_aligned = True
            for pt in [p1, p2]:
                vec_to_pt = pt - p0
                cross = np.abs(np.cross(vec_to_pt, line_vec))
                dist_sq = (cross / line_len) ** 2
                if dist_sq > tolerance_sq:
                    all_aligned = False
                    break

            if not all_aligned:
                continue

            # Valid bifold door found
            all_line_indices = set(v1['line_indices'] + v2['line_indices'])
            all_points = []
            for line_idx in all_line_indices:
                line = lines[line_idx]
                all_points.extend([line['start'], line['end']])

            xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            bifold_doors.append({
                'type': 'bifold_door',
                'v_candidates': [
                    {'joint_point': tuple(v1['joint_point']), 'angle': v1['angle'],
                     'avg_length': v1['avg_length'], 'line_indices': v1['line_indices']},
                    {'joint_point': tuple(v2['joint_point']), 'angle': v2['angle'],
                     'avg_length': v2['avg_length'], 'line_indices': v2['line_indices']}
                ],
                'line_ids': list(all_line_indices),
                'bbox': bbox
            })

            used_v_indices.add(i)
            used_v_indices.add(j)
            break

    if debug:
        for door in bifold_doors:
            bbox = door.get('bbox', (0, 0, 0, 0))
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            print(f"Bifold door: center=({int(center[0])}, {int(center[1])})")
        if bifold_doors:
            print(f"Found {len(bifold_doors)} bifold door(s)")

    return bifold_doors
