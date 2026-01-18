"""Door candidate generation from geometry."""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional


# ============================================================================
# UNIVERSAL  FUNCTIONS
# ============================================================================

def get_bezier_radius(control_points: List) -> Optional[Tuple[float, np.ndarray]]:
    """
    Calculate radius and midpoint of circular arc approximated by cubic Bezier.
    Returns (radius, midpoint) or None if points are collinear.
    """
    if len(control_points) != 4:
        return None

    p0, p1, p2, p3 = [np.array(p) for p in control_points]

    # Calculate midpoint curve
    t = 0.5
    mid = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

    # Calculate circumradius of triangle (p0, mid, p3)
    a = np.linalg.norm(mid - p0)
    b = np.linalg.norm(p3 - mid)
    c = np.linalg.norm(p3 - p0)

    # Area using cross product
    x1, y1 = p0
    x2, y2 = mid
    x3, y3 = p3
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    if area < 1e-5:
        return None  # Collinear points because triangle forms infinite area

    radius = (a * b * c) / (4 * area)
    return (radius, mid)


# ============================================================================
# ARC RECONSTRUCTION
# ============================================================================

class ArcReconstructor:
    """Reconstructs  arcs from fragmented line segments"""

    def __init__(self, page_width: float, page_height: float, debug: bool = False):
        self.page_width = page_width
        self.page_height = page_height
        self.debug = debug
        page_diagonal = np.sqrt(page_width**2 + page_height**2)
        self.segment_max_threshold = page_diagonal * 0.003
        self.segment_min_threshold = page_diagonal * 0.00035
        self.gap_tolerance = page_diagonal * 0.0009

    def _is_short_segment(self, line: Dict) -> bool:
        """Check if line is short enough to be a tessellated segment"""
        start, end = line['start'], line['end']
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx*dx + dy*dy) ** 0.5
        return self.segment_min_threshold <= length < self.segment_max_threshold

    def _squared_distance(self, p1: tuple, p2: tuple) -> float:
        """squared distance between two points"""
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx*dx + dy*dy

    def _are_connected(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines are connected within tolerance"""
        threshold_sq = self.gap_tolerance * self.gap_tolerance
        for p1 in [line1['start'], line1['end']]:
            for p2 in [line2['start'], line2['end']]:
                if self._squared_distance(p1, p2) <= threshold_sq:
                    return True
        return False

    def _continues_smoothly(self, chain: List[Tuple[int, Dict]], new_segment: Dict, add_to_end: bool) -> bool:
        """Check if new segment continues chain direction smoothly (angle <= 40°)"""
        if len(chain) == 0:
            return True

        if add_to_end:
            last_seg = chain[-1][1]
            last_start, last_end = np.array(
                last_seg['start']), np.array(last_seg['end'])
            chain_dir = last_end - last_start
            connect_pt = last_end
        else:
            first_seg = chain[0][1]
            first_start, first_end = np.array(
                first_seg['start']), np.array(first_seg['end'])
            chain_dir = first_end - first_start
            connect_pt = first_start

        new_start, new_end = np.array(
            new_segment['start']), np.array(new_segment['end'])
        dist_to_start_sq = np.sum((connect_pt - new_start) ** 2)
        dist_to_end_sq = np.sum((connect_pt - new_end) ** 2)

        if add_to_end:
            new_dir = (
                new_end - new_start) if dist_to_start_sq <= dist_to_end_sq else (new_start - new_end)
        else:
            new_dir = (
                new_end - new_start) if dist_to_start_sq <= dist_to_end_sq else (new_start - new_end)

        chain_norm = np.linalg.norm(chain_dir)
        new_norm = np.linalg.norm(new_dir)

        if chain_norm < 1e-5 or new_norm < 1e-5:
            return True

        chain_dir_unit = chain_dir / chain_norm
        new_dir_unit = new_dir / new_norm

        dot_product = np.clip(np.dot(chain_dir_unit, new_dir_unit), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_product))

        return angle_deg <= 40.0

    def _would_reverse_curve(self, chain: List[Tuple[int, Dict]], new_segment: Dict, add_to_end: bool) -> bool:
        """Check if adding new segment would reverse curve direction (>110°)"""
        if len(chain) < 2:
            return False

        if add_to_end:
            last_seg, prev_seg = chain[-1][1], chain[-2][1]
            last_dir = np.array(last_seg['end']) - np.array(last_seg['start'])
            prev_dir = np.array(prev_seg['end']) - np.array(prev_seg['start'])
            curve_dir = last_dir + prev_dir
            connect_pt = np.array(last_seg['end'])
        else:
            first_seg, second_seg = chain[0][1], chain[1][1]
            first_dir = np.array(
                first_seg['end']) - np.array(first_seg['start'])
            second_dir = np.array(
                second_seg['end']) - np.array(second_seg['start'])
            curve_dir = first_dir + second_dir
            connect_pt = np.array(first_seg['start'])

        new_s, new_e = np.array(
            new_segment['start']), np.array(new_segment['end'])
        dist_s_sq = np.sum((connect_pt - new_s) ** 2)
        dist_e_sq = np.sum((connect_pt - new_e) ** 2)

        if add_to_end:
            new_dir = (
                new_e - connect_pt) if dist_s_sq <= dist_e_sq else (new_s - connect_pt)
        else:
            new_dir = (
                connect_pt - new_e) if dist_s_sq <= dist_e_sq else (connect_pt - new_s)

        curve_norm, new_norm = np.linalg.norm(
            curve_dir), np.linalg.norm(new_dir)
        if curve_norm <= 1e-5 or new_norm <= 1e-5:
            return False

        dot = np.dot(curve_dir / curve_norm, new_dir / new_norm)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return angle > 110

    def _chain_segments(self, segments: List[Tuple[int, Dict]]) -> List[List[Tuple[int, Dict]]]:
        """Group chains using spatial indexing"""
        if not segments:
            return []

        # Build spatial index
        cell_size = self.gap_tolerance * 0.33
        spatial_index = {}

        def get_cell_key(point):
            return (int(point[0] / cell_size), int(point[1] / cell_size))

        for i, (orig_idx, seg) in enumerate(segments):
            start_key = get_cell_key(seg['start'])
            end_key = get_cell_key(seg['end'])
            midpoint = ((seg['start'][0] + seg['end'][0]) / 2,
                        (seg['start'][1] + seg['end'][1]) / 2)
            mid_key = get_cell_key(midpoint)

            for key in [start_key, end_key, mid_key]:
                if key not in spatial_index:
                    spatial_index[key] = []
                spatial_index[key].append(i)

        def get_candidate_indices(segment):
            """Get candidate segments using spatial index"""
            candidates = set()
            gap_tolerance_sq = self.gap_tolerance * self.gap_tolerance

            for point in [segment['start'], segment['end']]:
                cell_key = get_cell_key(point)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_key = (cell_key[0] + dx, cell_key[1] + dy)
                        if neighbor_key in spatial_index:
                            for candidate_idx in spatial_index[neighbor_key]:
                                candidate_seg = segments[candidate_idx][1]
                                min_dist_sq = float('inf')
                                for p2 in [candidate_seg['start'], candidate_seg['end']]:
                                    dist_sq = self._squared_distance(point, p2)
                                    min_dist_sq = min(min_dist_sq, dist_sq)
                                if min_dist_sq <= gap_tolerance_sq:
                                    candidates.add(candidate_idx)
            return candidates

        chains = []
        used = set()

        for i, (orig_idx, seg) in enumerate(segments):
            if i in used:
                continue

            chain = [(orig_idx, seg)]
            used.add(i)
            changed = True
            fallback_used = False

            while changed:
                changed = False
                candidates = get_candidate_indices(
                    chain[-1][1]) | get_candidate_indices(chain[0][1])

                # Fallback: check all segments if spatial index finds few candidates
                if not fallback_used and len(candidates) < 10 and len(chain) < 20:
                    candidates = set(range(len(segments))) - used
                    fallback_used = True

                segments_to_add_end = []
                segments_to_add_start = []

                for j in candidates:
                    if j in used or j >= len(segments):
                        continue

                    other_orig_idx, other_seg = segments[j]

                    if self._are_connected(chain[-1][1], other_seg):
                        if not self._would_reverse_curve(chain, other_seg, add_to_end=True):
                            if self._continues_smoothly(chain, other_seg, add_to_end=True):
                                segments_to_add_end.append(
                                    (j, other_orig_idx, other_seg))
                    elif self._are_connected(chain[0][1], other_seg):
                        if not self._would_reverse_curve(chain, other_seg, add_to_end=False):
                            if self._continues_smoothly(chain, other_seg, add_to_end=False):
                                segments_to_add_start.append(
                                    (j, other_orig_idx, other_seg))

                for j, other_orig_idx, other_seg in segments_to_add_end:
                    if j not in used:
                        chain.append((other_orig_idx, other_seg))
                        used.add(j)
                        changed = True

                for j, other_orig_idx, other_seg in reversed(segments_to_add_start):
                    if j not in used:
                        chain.insert(0, (other_orig_idx, other_seg))
                        used.add(j)
                        changed = True

            if len(chain) >= 3:
                chains.append(chain)

        return chains

    def _collect_chain_points(self, chain: List[Tuple[int, Dict]]) -> List[np.ndarray]:
        """Collect ordered points from a chain of segments"""
        points = []
        current_point = None

        for orig_idx, line in chain:
            start, end = np.array(line['start']), np.array(line['end'])

            if len(points) == 0:
                points.extend([start, end])
                current_point = end
            else:
                current_arr = np.array(current_point)
                gap_tol_sq = self.gap_tolerance * self.gap_tolerance
                dist_to_start_sq = np.sum((start - current_arr) ** 2)
                dist_to_end_sq = np.sum((end - current_arr) ** 2)

                if dist_to_start_sq <= dist_to_end_sq and dist_to_start_sq <= gap_tol_sq:
                    points.append(end)
                    current_point = end
                elif dist_to_end_sq <= gap_tol_sq:
                    points.append(start)
                    current_point = start
                else:
                    points.extend([start, end])
                    current_point = end

        return points

    def _calculate_detour_index(self, chain: List[Tuple[int, Dict]]) -> float:
        """ detour index: Total Path Length / Straight Distance"""
        if len(chain) < 2:
            return 1.0

        points = self._collect_chain_points(chain)
        if len(points) < 2:
            return 1.0

        total_length = sum(np.linalg.norm(
            points[i+1] - points[i]) for i in range(len(points) - 1))
        straight_distance = np.linalg.norm(points[-1] - points[0])

        return total_length / straight_distance if straight_distance >= 1e-5 else 1.0

    def _fit_circle(self, chain: List[Tuple[int, Dict]]) -> Tuple[Optional[Dict], Optional[str], Optional[Dict]]:
        """Fit circle to chain of segments returns (arc_dict, diagnostic_string, metrics_dict)"""
        points = self._collect_chain_points(chain)

        if len(points) < 3:
            return None, f"insufficient_points({len(points)})", None

        points_array = np.array(points)
        p0 = points_array[0]
        p_mid = points_array[len(points_array) // 2]
        p_end = points_array[-1]

        try:
            # Calculate circle center from perpendicular bisectors
            v1, v2 = p_mid - p0, p_end - p0
            mid1, mid2 = (p0 + p_mid) / 2, (p0 + p_end) / 2
            perp1, perp2 = np.array([-v1[1], v1[0]]), np.array([-v2[1], v2[0]])

            A = np.column_stack([perp1, -perp2])
            b = mid2 - mid1

            if np.linalg.det(A) == 0:
                return None, "collinear_points", None

            t_s = np.linalg.solve(A, b)
            center = mid1 + t_s[0] * perp1
            radius = np.linalg.norm(p0 - center)

            # Check max error
            max_error = max(np.abs(np.linalg.norm(p - center) - radius)
                            for p in points_array)
            error_threshold = radius * 0.8

            metrics = {'radius': radius, 'max_error': max_error,
                       'error_threshold': error_threshold}

            if max_error > error_threshold:
                return None, f"max_error_too_high({max_error:.2f} > {error_threshold:.2f})", metrics

            # Calculate sweep angle
            start_vec, end_vec = p0 - center, p_end - center
            start_angle = np.arctan2(start_vec[1], start_vec[0]) % (2 * np.pi)
            end_angle = np.arctan2(end_vec[1], end_vec[0]) % (2 * np.pi)

            diff = end_angle - start_angle
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi

            sweep_angle = abs(diff)
            sweep_angle_deg = np.degrees(sweep_angle)
            metrics['sweep_angle_deg'] = sweep_angle_deg

            # Validate sweep angle
            if sweep_angle_deg >= 360:
                return None, f"sweep_angle_out_of_range({sweep_angle_deg:.1f}° >= 360°)", metrics
            if sweep_angle_deg < 12:
                return None, f"sweep_angle_out_of_range({sweep_angle_deg:.1f}° < 12°)", metrics
            if sweep_angle_deg > 210:
                return None, f"sweep_angle_too_large({sweep_angle_deg:.1f}° > 210°)", metrics

            chord_length = np.linalg.norm(p_end - p0)
            if chord_length < 1e-1:
                return None, "chord_length_too_small", metrics

            arc_length = radius * sweep_angle
            arc_chord_ratio = arc_length / chord_length if chord_length > 0 else 0
            metrics.update({'chord_length': chord_length,
                           'arc_length': arc_length, 'arc_chord_ratio': arc_chord_ratio})

            if arc_length <= chord_length * 1.01:
                return None, f"arc_too_flat(ratio={arc_chord_ratio:.4f} <= 1.01)", metrics

            # Validate radius
            page_diagonal = np.sqrt(self.page_width**2 + self.page_height**2)
            min_radius = page_diagonal * 0.0015
            max_radius = page_diagonal * 0.1
            metrics.update({'page_diagonal': page_diagonal,
                           'min_radius': min_radius, 'max_radius': max_radius})

            if radius < min_radius or radius > max_radius:
                margin = min_radius - radius if radius < min_radius else radius - max_radius
                return None, f"radius_out_of_range({radius:.2f}, valid: {min_radius:.2f}-{max_radius:.2f})", metrics

            chord_radius_ratio = chord_length / radius if radius > 0 else 0
            metrics['chord_radius_ratio'] = chord_radius_ratio

            if chord_radius_ratio < 0.3 or chord_radius_ratio > 3.2:
                margin = 0.3 - chord_radius_ratio if chord_radius_ratio < 0.3 else chord_radius_ratio - 3.2
                return None, f"chord_radius_ratio_out_of_range({chord_radius_ratio:.2f}, valid: 0.3-3.2)", metrics

            # Calculate tangent directions
            start_tangent = np.array(
                [-np.sin(start_angle), np.cos(start_angle)])
            end_tangent = np.array([-np.sin(end_angle), np.cos(end_angle)])

            # For near-180° arcs, use chain direction to determine orientation
            if len(points_array) >= 2:
                chain_dir = points_array[1] - points_array[0]
                chain_dir_norm = chain_dir / \
                    np.linalg.norm(chain_dir) if np.linalg.norm(
                        chain_dir) > 1e-5 else None

                if chain_dir_norm is not None:
                    start_tangent_norm = start_tangent / \
                        np.linalg.norm(start_tangent) if np.linalg.norm(
                            start_tangent) > 1e-5 else start_tangent
                    tangent_chain_dot = np.dot(
                        start_tangent_norm, chain_dir_norm)

                    if tangent_chain_dot < 0:
                        start_tangent = -start_tangent
                        end_tangent = -end_tangent
                        diff = -diff

            if sweep_angle < 1e-5:
                return None, "sweep_angle_too_small", metrics

            # Calculate control points for Bezier curve
            control_distance = radius * (4.0 / 3.0) * np.tan(sweep_angle / 4.0)
            p1 = p0 + control_distance * start_tangent
            p2 = p_end - control_distance * end_tangent

            if np.linalg.norm(p1 - p0) < 1e-5 or np.linalg.norm(p2 - p_end) < 1e-5:
                return None, "control_point_too_close_to_endpoint", metrics
            if np.linalg.norm(p1 - p2) < 1e-5:
                return None, "control_points_too_close", metrics

            stroke_width = chain[0][1].get('stroke_width', 1.0)
            all_control_points = [p0, p1, p2, p_end]
            all_x = [p[0] for p in all_control_points]
            all_y = [p[1] for p in all_control_points]
            path_rect = (float(min(all_x)), float(min(all_y)),
                         float(max(all_x)), float(max(all_y)))
            control_points_list = [[float(p[0]), float(p[1])]
                                   for p in all_control_points]

            return {
                'type': 'cubic_bezier',
                'control_points': control_points_list,
                'stroke_width': float(stroke_width),
                'path_rect': path_rect,
                'close_path': False,
                'reconstructed': True,
                'center': [float(center[0]), float(center[1])],
                'radius': float(radius),
                'sweep_angle': float(sweep_angle_deg)
            }, None, metrics
        except Exception as e:
            return None, f"exception: {str(e)}", None

    def reconstruct_arcs(self, lines: List[Dict]) -> Tuple[List[Dict], set, List[Dict]]:
        """
        Reconstruct arcs from tessellated line segments.
        Returns:
            Tuple of (reconstructed_arcs, used_line_indices, rejected_chains)
        """
        start_time = time.time() if self.debug else None

        segments_with_indices = [(i, line) for i, line in enumerate(
            lines) if self._is_short_segment(line)]

        if len(segments_with_indices) < 3:
            return [], set(), []

        chains = self._chain_segments(segments_with_indices)

        if not chains:
            return [], set(), []

        # Fit circles to chains
        reconstructed_arcs = []
        used_line_indices = set()
        rejected_chains = []

        def _get_chain_bbox_and_center(chain):
            all_x, all_y = [], []
            for orig_idx, line in chain:
                all_x.extend([line['start'][0], line['end'][0]])
                all_y.extend([line['start'][1], line['end'][1]])
            bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            return bbox, center

        for chain_idx, chain in enumerate(chains):
            chain_bbox, chain_center = _get_chain_bbox_and_center(chain)
            detour_index = self._calculate_detour_index(chain)

            if detour_index > 1.01:
                arc, diagnostic, metrics = self._fit_circle(chain)
                if arc is not None:
                    reconstructed_arcs.append(arc)
                    used_line_indices.update(orig_idx for orig_idx, _ in chain)
                else:
                    rejected_chains.append({
                        'chain_idx': chain_idx,
                        'bbox': chain_bbox,
                        'center': chain_center,
                        'reason': diagnostic,
                        'detour_index': detour_index,
                        'metrics': metrics
                    })
            else:
                rejected_chains.append({
                    'chain_idx': chain_idx,
                    'bbox': chain_bbox,
                    'center': chain_center,
                    'reason': f'detour_too_low({detour_index:.4f} <= 1.01)',
                    'detour_index': detour_index,
                    'metrics': None
                })

        if self.debug:
            print(
                f"Reconstructed {len(reconstructed_arcs)} arcs from {len(used_line_indices)} segments")

        return reconstructed_arcs, used_line_indices, rejected_chains


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def _filter_circular_annotation_patterns(arcs: List[Dict], page_width: float, page_height: float) -> List[Dict]:
    """Filter out groups of arcs that form circular annotation patterns."""
    if len(arcs) < 2:
        return arcs

    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    touch_tolerance_sq = (page_diagonal * 0.0023) ** 2

    # data we need later for arc
    arc_data = []
    for arc in arcs:
        cp = arc.get('control_points')
        if not cp or len(cp) != 4:
            arc_data.append(None)
            continue

        start, end = np.array(cp[0]), np.array(cp[3])
        result = get_bezier_radius(cp)
        if result is None:
            arc_data.append(None)
            continue

        radius, center = result
        sweep = arc.get('sweep_angle')
        if sweep is None:
            from src.door_classifier import calculate_arc_sweep_angle
            sweep = calculate_arc_sweep_angle(arc, radius)
            if sweep is None:
                arc_data.append(None)
                continue

        arc_data.append({'start': start, 'end': end,
                        'sweep': sweep, 'radius': radius, 'center': center})

    #  adjacency list
    n = len(arcs)
    adjacency = [[] for _ in range(n)]

    for i in range(n):
        if arc_data[i] is None:
            continue
        s1, e1 = arc_data[i]['start'], arc_data[i]['end']

        for j in range(i + 1, n):
            if arc_data[j] is None:
                continue
            s2, e2 = arc_data[j]['start'], arc_data[j]['end']

            touches = False
            for p1 in [s1, e1]:
                for p2 in [s2, e2]:
                    if np.sum((p1 - p2) ** 2) <= touch_tolerance_sq:
                        touches = True
                        break
                if touches:
                    break

            if touches:
                adjacency[i].append(j)
                adjacency[j].append(i)

    #  connected components using BFS
    visited = set()
    arc_indices_to_remove = set()

    for start_idx in range(n):
        if start_idx in visited or arc_data[start_idx] is None:
            continue

        component = []
        queue = [start_idx]
        visited.add(start_idx)

        while queue:
            idx = queue.pop(0)
            component.append(idx)
            for neighbor in adjacency[idx]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check if component matches pattern (2+ arcs, >230° total, similar centers/radii)
        if len(component) >= 2:
            total_sweep = sum(arc_data[idx]['sweep']
                              for idx in component if arc_data[idx] is not None)

            if total_sweep > 230:
                centers = [arc_data[idx]['center']
                           for idx in component if arc_data[idx] is not None]
                radii = [arc_data[idx]['radius']
                         for idx in component if arc_data[idx] is not None]

                if len(centers) >= 2 and len(radii) >= 2:
                    centers_array = np.array(centers)
                    center_mean = np.mean(centers_array, axis=0)
                    max_center_deviation = max(np.linalg.norm(
                        c - center_mean) for c in centers)

                    radius_mean = np.mean(radii)
                    max_radius_deviation = max(
                        abs(r - radius_mean) for r in radii)

                    center_tolerance = page_diagonal * 0.014
                    radius_tolerance = radius_mean * 0.17

                    if max_center_deviation < center_tolerance and max_radius_deviation < radius_tolerance:
                        arc_indices_to_remove.update(component)

    if arc_indices_to_remove:
        return [arc for idx, arc in enumerate(arcs) if idx not in arc_indices_to_remove]

    return arcs


def filter_door_candidates(lines: List[Dict], arcs: List[Dict], page_width: float, page_height: float, debug: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter door candidates by stroke width.
    Removes dust (very short) and extremely long lines/arcs before calculating percentiles
    """
    if not lines and not arcs:
        return lines, arcs

    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    MIN_LENGTH = page_diagonal * 0.0038
    MAX_LENGTH = page_diagonal * 0.04

    def line_length(line):
        start, end = line['start'], line['end']
        dx, dy = end[0] - start[0], end[1] - start[1]
        return (dx*dx + dy*dy) ** 0.5

    # Filter by length
    filtered_lines = [line for line in lines if MIN_LENGTH <=
                      line_length(line) <= MAX_LENGTH]

    filtered_arcs_for_percentile = []
    for arc in arcs:
        cp = arc.get('control_points', [])
        if len(cp) == 4:
            chord_length = np.linalg.norm(np.array(cp[3]) - np.array(cp[0]))
            if MIN_LENGTH <= chord_length <= MAX_LENGTH:
                filtered_arcs_for_percentile.append(arc)

    # Filter circular annotation patterns
    filtered_arcs_for_percentile = _filter_circular_annotation_patterns(
        filtered_arcs_for_percentile, page_width, page_height)

    # Calculate stroke width thresholds
    door_min_percentile, door_max_percentile = 30, 100
    line_strokes = [l['stroke_width'] for l in filtered_lines]
    arc_strokes = [a['stroke_width'] for a in filtered_arcs_for_percentile]
    all_strokes = line_strokes + arc_strokes

    if not all_strokes:
        return [], []

    min_threshold = np.percentile(all_strokes, door_min_percentile)
    max_threshold = np.percentile(all_strokes, door_max_percentile)

    # Arc-specific thresholds
    if arc_strokes:
        arc_min_threshold = np.percentile(arc_strokes, 20)
        arc_max_threshold = np.percentile(arc_strokes, 100)
    else:
        arc_min_threshold = arc_max_threshold = 0

    # Filter by stroke width
    door_lines = [line for line in filtered_lines if min_threshold <=
                  line['stroke_width'] <= max_threshold]
    door_arcs = [arc for arc in filtered_arcs_for_percentile if arc_min_threshold <=
                 arc['stroke_width'] <= arc_max_threshold]

    return door_lines, door_arcs


# ============================================================================
#  GEOMETRY ANALYSIS
# ============================================================================

def analyze_geometry(lines: List[Dict], arcs: List[Dict], dashed_lines: List[Dict], page_width: float, page_height: float, debug: bool = False) -> Dict:
    """
    Analyze geometry to find door candidates
    Returns dictionary with filtered lines, arcs, and door candidates
    """
    if debug:
        print(
            f"Lines: {len(lines)}, Arcs: {len(arcs)}, Dashed lines: {len(dashed_lines)}")

    all_lines = lines + dashed_lines

    reconstructor = ArcReconstructor(page_width, page_height, debug=debug)
    reconstructed_arcs, used_line_indices, rejected_chains = reconstructor.reconstruct_arcs(
        all_lines)

    if reconstructed_arcs:
        arcs = arcs + reconstructed_arcs
        used_set = used_line_indices if isinstance(
            used_line_indices, set) else set(used_line_indices)
        all_lines = [line for i, line in enumerate(
            all_lines) if i not in used_set]

    filtered_lines, filtered_arcs = filter_door_candidates(
        all_lines, arcs, page_width, page_height, debug=debug)

    if debug:
        print(
            f"Filtered lines: {len(filtered_lines)}, Filtered arcs: {len(filtered_arcs)}")

    swing_door_arcs = []
    double_door_candidates = []

    for arc in filtered_arcs:
        sweep_angle = arc.get('sweep_angle')
        if sweep_angle is not None:
            if 150 <= sweep_angle <= 210:
                double_door_candidates.append(arc)
            elif sweep_angle < 120:
                swing_door_arcs.append(arc)
        else:
            swing_door_arcs.append(arc)

    double_door_ids = {id(arc) for arc in double_door_candidates}
    filtered_arcs_without_double_doors = [
        arc for arc in filtered_arcs if id(arc) not in double_door_ids]

    if debug:
        print(
            f"Swing door arcs: {len(swing_door_arcs)}, Double door arcs: {len(double_door_candidates)}")

    return {
        "filtered_lines": filtered_lines,
        "filtered_arcs": filtered_arcs_without_double_doors,
        "door_candidate_arcs": swing_door_arcs,
        "dashed_lines": dashed_lines,
        "rejected_chains": rejected_chains,
        "double_door_candidates": double_door_candidates
    }
