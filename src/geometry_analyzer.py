"""Door candidate generation from geometry."""
import numpy as np
import time
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


class ArcReconstructor:
    """Reconstructs tessellated arcs from fragmented line segments."""

    def __init__(self, page_width: float, page_height: float, debug: bool = False):
        self.page_width = page_width
        self.page_height = page_height
        self.debug = debug
        page_diagonal = np.sqrt(page_width**2 + page_height**2)
        self.segment_max_threshold = page_diagonal * 0.003
        self.segment_min_threshold = page_diagonal * 0.00035
        self.gap_tolerance = page_diagonal * 0.0009

    def _is_short_segment(self, line: Dict) -> bool:
        """Check if line is short enough to be a tessellated segment (but not too small - dust)."""
        start = line['start']
        end = line['end']
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx*dx + dy*dy) ** 0.5
        return self.segment_min_threshold <= length < self.segment_max_threshold

    def _squared_distance(self, p1: tuple, p2: tuple) -> float:
        """Calculate squared distance between two points (avoids sqrt for speed)."""
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx*dx + dy*dy

    def _are_connected(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines are connected within tolerance."""
        threshold_sq = self.gap_tolerance * self.gap_tolerance
        endpoints1 = [line1['start'], line1['end']]
        endpoints2 = [line2['start'], line2['end']]

        # Check all 4 endpoint combinations
        for p1 in endpoints1:
            for p2 in endpoints2:
                if self._squared_distance(p1, p2) <= threshold_sq:
                    return True
        return False

    def _continues_smoothly(self, chain: List[Tuple[int, Dict]], new_segment: Dict, add_to_end: bool) -> bool:
        """
        Check if a new segment continues the chain direction smoothly (not an intersecting line).

        Args:
            chain: Current chain of segments
            new_segment: Segment to potentially add
            add_to_end: True if adding to end of chain, False if adding to start

        Returns:
            True if the segment continues smoothly (angle <= 40 degrees), False otherwise
        """
        if len(chain) == 0:
            return True  # First segment always allowed

        if add_to_end:
            # Get the direction of the chain at the end (outward from connection point)
            last_seg = chain[-1][1]
            last_start = np.array(last_seg['start'])
            last_end = np.array(last_seg['end'])
            # Chain direction is from start to end of last segment (outward from connection)
            chain_dir = last_end - last_start

            # Find which endpoint of new_segment connects to last_end
            new_start = np.array(new_segment['start'])
            new_end = np.array(new_segment['end'])
            # Use squared distances for comparison (avoid sqrt)
            dist_to_start_sq = np.sum((last_end - new_start) ** 2)
            dist_to_end_sq = np.sum((last_end - new_end) ** 2)

            if dist_to_start_sq <= dist_to_end_sq:
                # new_segment connects at its start, so direction is start->end (continuing forward)
                new_dir = new_end - new_start
            else:
                # new_segment connects at its end, so direction is end->start (reversed)
                new_dir = new_start - new_end
        else:
            # Get the direction of the chain at the start (outward from connection point)
            first_seg = chain[0][1]
            first_start = np.array(first_seg['start'])
            first_end = np.array(first_seg['end'])
            # Chain direction is from start to end of first segment (outward from connection)
            chain_dir = first_end - first_start

            # Find which endpoint of new_segment connects to first_start
            new_start = np.array(new_segment['start'])
            new_end = np.array(new_segment['end'])
            # Use squared distances for comparison (avoid sqrt)
            dist_to_start_sq = np.sum((first_start - new_start) ** 2)
            dist_to_end_sq = np.sum((first_start - new_end) ** 2)

            if dist_to_start_sq <= dist_to_end_sq:
                # new_segment connects at its start, so direction outward is start->end
                new_dir = new_end - new_start
            else:
                # new_segment connects at its end, so direction outward is end->start (reversed)
                new_dir = new_start - new_end

        # Normalize direction vectors
        chain_norm = np.linalg.norm(chain_dir)
        new_norm = np.linalg.norm(new_dir)

        if chain_norm < 1e-5 or new_norm < 1e-5:
            return True  # Degenerate case, allow it

        chain_dir_unit = chain_dir / chain_norm
        new_dir_unit = new_dir / new_norm

        # Calculate angle between directions using dot product
        dot_product = np.clip(np.dot(chain_dir_unit, new_dir_unit), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        # Allow angles up to 40 degrees (smooth continuation for arcs)
        # Reject sharp turns (>40 degrees) which indicate intersecting lines
        return angle_deg <= 40.0

    def _would_reverse_curve(self, chain: List[Tuple[int, Dict]], new_segment: Dict, add_to_end: bool) -> bool:
        """Check if adding new_segment would significantly reverse curve direction (>100°)."""
        if len(chain) < 2:
            return False

        if add_to_end:
            last_seg = chain[-1][1]
            prev_seg = chain[-2][1]
            last_dir = np.array(last_seg['end']) - np.array(last_seg['start'])
            prev_dir = np.array(prev_seg['end']) - np.array(prev_seg['start'])
            curve_dir = last_dir + prev_dir
            connect_pt = np.array(last_seg['end'])
        else:
            first_seg = chain[0][1]
            second_seg = chain[1][1]
            first_dir = np.array(
                first_seg['end']) - np.array(first_seg['start'])
            second_dir = np.array(
                second_seg['end']) - np.array(second_seg['start'])
            curve_dir = first_dir + second_dir
            connect_pt = np.array(first_seg['start'])

        new_s, new_e = np.array(
            new_segment['start']), np.array(new_segment['end'])
        # Use squared distances for comparison (avoid sqrt)
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
        """Group connected segments into chains, avoiding intersecting lines.

        Optimized with spatial indexing to reduce time complexity from O(n²) to ~O(n*k)
        where k is the average number of segments in nearby spatial cells.
        Uses lenient spatial indexing with fallback to ensure no connections are missed.
        """
        if not segments:
            return []

        # Build spatial index: map grid cell -> list of segment indices
        # Use cell_size = gap_tolerance * 0.75 with 2x2 neighborhood
        # This limits candidates to ~1.06 * gap_tolerance away (tight control)
        cell_size = self.gap_tolerance * .33
        spatial_index = {}

        def get_cell_key(point):
            """Get spatial grid cell key for a point."""
            return (int(point[0] / cell_size), int(point[1] / cell_size))

        # Index all segment endpoints AND midpoints for better coverage
        for i, (orig_idx, seg) in enumerate(segments):
            start_key = get_cell_key(seg['start'])
            end_key = get_cell_key(seg['end'])
            # Also index midpoint to catch segments that might connect
            midpoint = ((seg['start'][0] + seg['end'][0]) / 2,
                        (seg['start'][1] + seg['end'][1]) / 2)
            mid_key = get_cell_key(midpoint)

            for key in [start_key, end_key, mid_key]:
                if key not in spatial_index:
                    spatial_index[key] = []
                spatial_index[key].append(i)

        def get_candidate_indices(segment):
            """Get candidate segment indices that might connect to this segment.
            Uses spatial index for speed, then filters by actual distance to ensure
            we find all segments within gap_tolerance regardless of cell size."""
            candidates = set()
            gap_tolerance_sq = self.gap_tolerance * self.gap_tolerance

            # Check cells for both endpoints with 3x3 neighborhood (larger to catch more)
            # We'll filter by distance anyway, so checking more cells is safe
            for point in [segment['start'], segment['end']]:
                cell_key = get_cell_key(point)
                # Check 3x3 neighborhood to ensure we don't miss segments
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_key = (cell_key[0] + dx, cell_key[1] + dy)
                        if neighbor_key in spatial_index:
                            # Filter by actual distance - only add if within gap_tolerance
                            for candidate_idx in spatial_index[neighbor_key]:
                                candidate_seg = segments[candidate_idx][1]
                                # Quick distance check: are any endpoints within gap_tolerance?
                                min_dist_sq = float('inf')
                                for p1 in [point]:
                                    for p2 in [candidate_seg['start'], candidate_seg['end']]:
                                        dist_sq = self._squared_distance(
                                            p1, p2)
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
            fallback_used = False  # Track if we've used fallback for this chain

            while changed:
                changed = False
                # Get candidates from spatial index
                candidates = get_candidate_indices(
                    chain[-1][1]) | get_candidate_indices(chain[0][1])

                # Safety fallback: if spatial index finds very few candidates and we haven't used fallback yet,
                # check all remaining segments once to ensure we don't miss connections
                # More lenient: trigger fallback more easily for better completeness
                if not fallback_used and len(candidates) < 10 and len(chain) < 20:
                    # Fallback: check all unused segments (slower but ensures completeness)
                    candidates = set(range(len(segments))) - used
                    fallback_used = True  # Only use fallback once per chain to balance speed/accuracy

                # Collect valid segments to add (check all candidates first, then add them)
                # This is less greedy - finds multiple connections per iteration
                segments_to_add_end = []
                segments_to_add_start = []

                for j in candidates:
                    if j in used or j >= len(segments):
                        continue

                    other_orig_idx, other_seg = segments[j]

                    connects_to_end = self._are_connected(
                        chain[-1][1], other_seg)
                    connects_to_start = self._are_connected(
                        chain[0][1], other_seg)

                    # Collect valid segments to add (check all candidates first, then add them)
                    if connects_to_end:
                        if not self._would_reverse_curve(chain, other_seg, add_to_end=True):
                            if self._continues_smoothly(chain, other_seg, add_to_end=True):
                                segments_to_add_end.append(
                                    (j, other_orig_idx, other_seg))
                    elif connects_to_start:
                        if not self._would_reverse_curve(chain, other_seg, add_to_end=False):
                            if self._continues_smoothly(chain, other_seg, add_to_end=False):
                                segments_to_add_start.append(
                                    (j, other_orig_idx, other_seg))

                # Add segments to end (in order found)
                for j, other_orig_idx, other_seg in segments_to_add_end:
                    if j not in used:  # Double-check in case of duplicates
                        chain.append((other_orig_idx, other_seg))
                        used.add(j)
                        changed = True

                # Add segments to start (in reverse order to maintain chain order)
                for j, other_orig_idx, other_seg in reversed(segments_to_add_start):
                    if j not in used:  # Double-check in case of duplicates
                        chain.insert(0, (other_orig_idx, other_seg))
                        used.add(j)
                        changed = True

                # Break and restart while loop with updated chain endpoints
                # This allows the algorithm to find connections from the newly added segments

            if len(chain) >= 3:
                chains.append(chain)

        return chains

    def _collect_chain_points(self, chain: List[Tuple[int, Dict]]) -> List[np.ndarray]:
        """Collect ordered points from a chain of segments."""
        points = []
        current_point = None

        for orig_idx, line in chain:
            start = np.array(line['start'])
            end = np.array(line['end'])

            if len(points) == 0:
                points.append(start)
                points.append(end)
                current_point = end
            else:
                current_arr = np.array(current_point)
                # Use squared distances to avoid sqrt
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
                    points.append(start)
                    points.append(end)
                    current_point = end

        return points

    def _calculate_detour_index(self, chain: List[Tuple[int, Dict]]) -> float:
        """Calculate detour index: Total Path Length / Straight Distance."""
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
        """Fit circle to chain of segments using geometric circle fitting.
        Returns (arc_dict, diagnostic_string, metrics_dict) where:
        - arc_dict is the arc on success, None on failure
        - diagnostic_string is None on success, or contains the rejection reason on failure
        - metrics_dict contains computed values (radius, sweep_angle, etc.) for debugging"""
        points = self._collect_chain_points(chain)

        if len(points) < 3:
            return None, f"insufficient_points({len(points)})", None

        points_array = np.array(points)
        p0 = points_array[0]
        p_mid = points_array[len(points_array) // 2]
        p_end = points_array[-1]

        try:
            v1 = p_mid - p0
            v2 = p_end - p0
            mid1 = (p0 + p_mid) / 2
            mid2 = (p0 + p_end) / 2
            perp1 = np.array([-v1[1], v1[0]])
            perp2 = np.array([-v2[1], v2[0]])

            A = np.column_stack([perp1, -perp2])
            b = mid2 - mid1

            if np.linalg.det(A) == 0:
                return None, "collinear_points", None

            t_s = np.linalg.solve(A, b)
            center = mid1 + t_s[0] * perp1
            radius = np.linalg.norm(p0 - center)

            max_error = 0
            for p in points_array:
                dist = np.abs(np.linalg.norm(p - center) - radius)
                max_error = max(max_error, dist)

            error_threshold = radius * 0.8
            metrics = {'radius': radius, 'max_error': max_error,
                       'error_threshold': error_threshold}

            if max_error > error_threshold:
                return None, f"max_error_too_high({max_error:.2f} > {error_threshold:.2f}, radius={radius:.2f}, margin={max_error - error_threshold:.2f})", metrics

            start_vec = p0 - center
            end_vec = p_end - center
            start_angle = np.arctan2(start_vec[1], start_vec[0])
            end_angle = np.arctan2(end_vec[1], end_vec[0])

            # Calculate sweep angle - ensure we get the smaller arc and correct direction
            start_angle = start_angle % (2 * np.pi)
            end_angle = end_angle % (2 * np.pi)

            diff = end_angle - start_angle
            if diff > np.pi:
                diff = diff - 2 * np.pi
            elif diff < -np.pi:
                diff = diff + 2 * np.pi

            sweep_angle = abs(diff)
            sweep_angle_deg = np.degrees(sweep_angle)
            metrics['sweep_angle_deg'] = sweep_angle_deg

            # Ensure arc is open (not closed) - allow up to 210° for double door detection
            if sweep_angle_deg >= 360:
                return None, f"sweep_angle_out_of_range({sweep_angle_deg:.1f}° >= 360°)", metrics
            if sweep_angle_deg < 12:
                return None, f"sweep_angle_out_of_range({sweep_angle_deg:.1f}° < 12°, margin={sweep_angle_deg - 12:.1f}°)", metrics
            # Allow arcs up to 210° for double door detection (150-210° range)
            if sweep_angle_deg > 210:
                return None, f"sweep_angle_too_large({sweep_angle_deg:.1f}° > 210°, margin={sweep_angle_deg - 210:.1f}°)", metrics

            chord_length = np.linalg.norm(p_end - p0)
            if chord_length < 1e-1:
                return None, "chord_length_too_small", metrics

            arc_length = radius * sweep_angle
            arc_chord_ratio = arc_length / chord_length if chord_length > 0 else 0
            metrics.update({
                'chord_length': chord_length,
                'arc_length': arc_length,
                'arc_chord_ratio': arc_chord_ratio
            })

            if arc_length <= chord_length * 1.01:
                return None, f"arc_too_flat(arc={arc_length:.2f}, chord={chord_length:.2f}, ratio={arc_chord_ratio:.4f}, need >1.01, margin={arc_chord_ratio - 1.01:.4f})", metrics

            page_diagonal = np.sqrt(self.page_width**2 + self.page_height**2)
            min_radius = page_diagonal * 0.0015
            max_radius = page_diagonal * 0.1
            metrics.update({
                'page_diagonal': page_diagonal,
                'min_radius': min_radius,
                'max_radius': max_radius
            })

            if radius < min_radius or radius > max_radius:
                margin = min_radius - radius if radius < min_radius else radius - max_radius
                return None, f"radius_out_of_range({radius:.2f}, valid: {min_radius:.2f}-{max_radius:.2f}, margin={margin:.2f})", metrics

            chord_radius_ratio = chord_length / radius if radius > 0 else 0
            metrics['chord_radius_ratio'] = chord_radius_ratio

            if chord_radius_ratio < 0.3 or chord_radius_ratio > 3.2:
                margin = 0.3 - chord_radius_ratio if chord_radius_ratio < 0.3 else chord_radius_ratio - 3.2
                return None, f"chord_radius_ratio_out_of_range({chord_radius_ratio:.2f}, valid: 0.3-3.2, margin={margin:.2f})", metrics

            # Calculate tangent directions (perpendicular to radius vectors)
            # Tangent = rotate radius 90°: [-sin(angle), cos(angle)] for CCW, [sin(angle), -cos(angle)] for CW
            start_tangent = np.array(
                [-np.sin(start_angle), np.cos(start_angle)])
            end_tangent = np.array([-np.sin(end_angle), np.cos(end_angle)])

            # For near-180° arcs, use actual chain direction to determine correct orientation
            # The angle-based approach (diff) can be ambiguous for wide arcs
            if len(points_array) >= 2:
                # Get actual direction the chain is moving (from first to second point)
                chain_dir = points_array[1] - points_array[0]
                chain_dir_norm = chain_dir / \
                    np.linalg.norm(chain_dir) if np.linalg.norm(
                        chain_dir) > 1e-5 else None

                if chain_dir_norm is not None:
                    # Normalize start_tangent for comparison
                    start_tangent_norm = start_tangent / \
                        np.linalg.norm(start_tangent) if np.linalg.norm(
                            start_tangent) > 1e-5 else start_tangent

                    # Check if tangent aligns with chain direction
                    # If dot product is negative, tangents point opposite to chain direction
                    tangent_chain_dot = np.dot(
                        start_tangent_norm, chain_dir_norm)

                    # For near-180° arcs, we need to check both possible orientations
                    # If the tangent doesn't align well with chain direction, flip it
                    if tangent_chain_dot < 0:
                        # Tangents point opposite to chain direction - flip them
                        start_tangent = -start_tangent
                        end_tangent = -end_tangent
                        # Also flip diff to match
                        diff = -diff
            else:
                # Fallback: use angle difference if chain direction unavailable
                if diff < 0:
                    start_tangent = -start_tangent
                    end_tangent = -end_tangent

            sweep_rad = sweep_angle
            if sweep_rad < 1e-5:
                return None, "sweep_angle_too_small", metrics

            control_distance = radius * (4.0 / 3.0) * np.tan(sweep_rad / 4.0)

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
        """Reconstruct arcs from tessellated line segments.

        Returns:
            Tuple of (reconstructed_arcs, used_line_indices, rejected_chains)
            where rejected_chains is a list of dicts with chain info for visualization
        """
        start_time = time.time() if self.debug else None

        segments_with_indices = []
        for i, line in enumerate(lines):
            if self._is_short_segment(line):
                segments_with_indices.append((i, line))

        segment_time = time.time() if self.debug else None
        if self.debug:
            print(
                f"DEBUG ArcReconstructor: Found {len(segments_with_indices)} candidate segments out of {len(lines)} total lines (took {segment_time - start_time:.3f}s)")

        if len(segments_with_indices) < 3:
            return [], set(), []

        chains = self._chain_segments(segments_with_indices)
        chain_time = time.time() if self.debug else None
        if self.debug:
            print(
                f"DEBUG ArcReconstructor: Formed {len(chains)} chains from segments (took {chain_time - segment_time:.3f}s)")

        if not chains:
            return [], set(), []

        reconstructed_arcs = []
        used_line_indices = set()
        rejected_detour = 0
        rejected_fit = 0
        fit_rejection_details = []
        rejected_chains = []  # For visualization

        def _get_chain_bbox_and_center(chain):
            """Calculate chain bounding box and center."""
            all_x = []
            all_y = []
            for orig_idx, line in chain:
                all_x.extend([line['start'][0], line['end'][0]])
                all_y.extend([line['start'][1], line['end'][1]])
            bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            return bbox, center

        for chain_idx, chain in enumerate(chains):
            chain_bbox, chain_center = _get_chain_bbox_and_center(chain)

            detour_index = self._calculate_detour_index(chain)
            if detour_index > 1.01:  # Stricter: increased from 1.01
                arc, diagnostic, metrics = self._fit_circle(chain)
                if arc is not None:
                    reconstructed_arcs.append(arc)
                    for orig_idx, _ in chain:
                        used_line_indices.add(orig_idx)
                    if self.debug:
                        print(
                            f"DEBUG ArcReconstructor: Chain {chain_idx}: Reconstructed arc (radius={arc['radius']:.2f}, sweep={arc['sweep_angle']:.1f}°, center=({chain_center[0]:.1f}, {chain_center[1]:.1f}))")
                else:
                    rejected_fit += 1
                    fit_rejection_details.append(
                        (chain_idx, detour_index, diagnostic, metrics, chain_center, chain_bbox))
                    rejected_chains.append({
                        'chain_idx': chain_idx,
                        'bbox': chain_bbox,
                        'center': chain_center,
                        'reason': diagnostic,
                        'detour_index': detour_index,
                        'metrics': metrics
                    })
                    if self.debug:
                        print(
                            f"DEBUG ArcReconstructor: Chain {chain_idx}: Rejected fit - detour={detour_index:.4f}, reason={diagnostic}, center=({chain_center[0]:.1f}, {chain_center[1]:.1f}), bbox=({chain_bbox[0]:.1f},{chain_bbox[1]:.1f})-({chain_bbox[2]:.1f},{chain_bbox[3]:.1f})")
            else:
                rejected_detour += 1
                rejected_chains.append({
                    'chain_idx': chain_idx,
                    'bbox': chain_bbox,
                    'center': chain_center,
                    'reason': f'detour_too_low({detour_index:.4f} <= 1.015)',
                    'detour_index': detour_index,
                    'metrics': None
                })
                if self.debug:
                    print(
                        f"DEBUG ArcReconstructor: Chain {chain_idx}: Rejected detour - detour_index={detour_index:.4f} (need >1.01), center=({chain_center[0]:.1f}, {chain_center[1]:.1f})")

        fit_time = time.time() if self.debug else None
        total_time = fit_time - start_time if self.debug else None

        if self.debug:
            print(
                f"DEBUG ArcReconstructor: Rejected {rejected_detour} chains (detour), {rejected_fit} chains (fit) (fitting took {fit_time - chain_time:.3f}s)")

            if fit_rejection_details:
                print(f"\nDEBUG ArcReconstructor: Fit rejection summary:")
                rejection_reasons = {}
                for _, _, reason, _, _, _ in fit_rejection_details:
                    reason_type = reason.split(
                        '(')[0] if '(' in reason else reason
                    rejection_reasons[reason_type] = rejection_reasons.get(
                        reason_type, 0) + 1
                for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {reason}: {count} chains")

                print(
                    f"\nDEBUG ArcReconstructor: Detailed rejection info (first 20 chains):")
                for chain_idx, detour, reason, metrics, center, bbox in fit_rejection_details[:20]:
                    print(
                        f"  Chain {chain_idx}: center=({center[0]:.1f}, {center[1]:.1f}), detour={detour:.4f}, reason={reason}")
                    if metrics:
                        radius = metrics.get('radius', 'N/A')
                        sweep = metrics.get('sweep_angle_deg', 'N/A')
                        ratio = metrics.get('chord_radius_ratio', 'N/A')
                        radius_str = f"{radius:.2f}" if isinstance(
                            radius, (int, float)) else radius
                        sweep_str = f"{sweep:.1f}°" if isinstance(
                            sweep, (int, float)) else sweep
                        ratio_str = f"{ratio:.2f}" if isinstance(
                            ratio, (int, float)) else ratio
                        print(
                            f"    Metrics: radius={radius_str}, sweep_angle={sweep_str}, chord_radius_ratio={ratio_str}")

            print(
                f"\nDEBUG ArcReconstructor: Final result: {len(reconstructed_arcs)} reconstructed arcs from {len(used_line_indices)} line segments")
            print(
                f"DEBUG ArcReconstructor: Total time taken: {total_time:.3f} seconds")
        return reconstructed_arcs, used_line_indices, rejected_chains


def _filter_circular_annotation_patterns(arcs: List[Dict], page_width: float, page_height: float) -> List[Dict]:
    """
    Filter out groups of arcs that form circular annotation patterns (2+ arcs forming >180° circle).
    Optimized: builds connectivity graph once, uses BFS to find all connected components.
    More aggressive: lower thresholds to catch more circular patterns.
    """
    if len(arcs) < 2:
        return arcs  # Need at least 2 arcs to form a circle pattern

    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    # Squared distance for speed - slightly more aggressive: larger tolerance
    touch_tolerance_sq = (page_diagonal * 0.0023) ** 2

    n = len(arcs)

    # Pre-compute endpoints, centers, radii, and sweep angles (cache for speed)
    arc_data = []
    for arc in arcs:
        cp = arc.get('control_points')
        if not cp or len(cp) != 4:
            arc_data.append(None)
            continue

        start = np.array(cp[0])
        end = np.array(cp[3])

        # Get radius and center (needed for pattern detection)
        result = get_bezier_radius(cp)
        if result is None:
            arc_data.append(None)
            continue
        radius, center = result

        # Get sweep angle (cache it)
        sweep = arc.get('sweep_angle')
        if sweep is None:
            from src.door_classifier import calculate_arc_sweep_angle
            sweep = calculate_arc_sweep_angle(arc, radius)
            if sweep is None:
                arc_data.append(None)
                continue

        arc_data.append({
            'start': start,
            'end': end,
            'sweep': sweep,
            'radius': radius,
            'center': center
        })

    # Build adjacency list (which arcs touch each other) - O(n²) but optimized
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        if arc_data[i] is None:
            continue
        s1, e1 = arc_data[i]['start'], arc_data[i]['end']

        for j in range(i + 1, n):
            if arc_data[j] is None:
                continue
            s2, e2 = arc_data[j]['start'], arc_data[j]['end']

            # Check if any endpoints are close (squared distance for speed)
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

    # Find all connected components using BFS
    visited = set()
    arc_indices_to_remove = set()

    for start_idx in range(n):
        if start_idx in visited or arc_data[start_idx] is None:
            continue

        # BFS to find all connected arcs
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

        # Check if this component matches the pattern (2+ arcs, >180° total)
        # BUT also verify they form a closed circle (similar centers and radii)
        # More aggressive: lowered from 3+ arcs to 2+ arcs, and from >200° to >180°
        if len(component) >= 2:
            total_sweep = sum(arc_data[idx]['sweep']
                              for idx in component if arc_data[idx] is not None)

            if total_sweep > 230:
                # Additional validation: check if arcs share similar centers and radii
                # (annotation patterns form closed circles with uniform geometry)
                centers = [arc_data[idx]['center']
                           for idx in component if arc_data[idx] is not None]
                radii = [arc_data[idx]['radius']
                         for idx in component if arc_data[idx] is not None]

                # More aggressive: lowered from 3 to 2
                if len(centers) >= 2 and len(radii) >= 2:
                    # Check if centers are clustered and radii are similar (annotation patterns)
                    centers_array = np.array(centers)
                    center_mean = np.mean(centers_array, axis=0)
                    max_center_deviation = max(np.linalg.norm(
                        c - center_mean) for c in centers)

                    radius_mean = np.mean(radii)
                    max_radius_deviation = max(
                        abs(r - radius_mean) for r in radii)

                    # More aggressive: slightly increased tolerances to catch more patterns
                    center_tolerance = page_diagonal * 0.014
                    radius_tolerance = radius_mean * 0.17

                    if max_center_deviation < center_tolerance and max_radius_deviation < radius_tolerance:
                        arc_indices_to_remove.update(component)

    if arc_indices_to_remove:
        filtered_arcs = [arc for idx, arc in enumerate(
            arcs) if idx not in arc_indices_to_remove]
        # Note: debug flag not passed to this function, but we can add it if needed
        return filtered_arcs

    return arcs


def filter_door_candidates(lines: List[Dict], arcs: List[Dict], page_width: float, page_height: float, debug: bool = False) -> Tuple[List[Dict], List[Dict]]:
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
    MIN_LENGTH = page_diagonal * 0.00375  # 0.5% of page diagonal (dust)
    MAX_LENGTH = page_diagonal * 0.04  # 6% of page diagonal (extremely long)

    def line_length(line):
        start, end = line['start'], line['end']
        dx, dy = end[0] - start[0], end[1] - start[1]
        return (dx*dx + dy*dy) ** 0.5

    # Filter out dust and extremely long lines
    filtered_lines = [line for line in lines if MIN_LENGTH <=
                      line_length(line) <= MAX_LENGTH]

    # Filter out dust and extremely long arcs (using chord length)
    filtered_arcs_for_percentile = []
    for arc in arcs:
        cp = arc.get('control_points', [])
        if len(cp) == 4:
            chord_length = np.linalg.norm(np.array(cp[3]) - np.array(cp[0]))
            if MIN_LENGTH <= chord_length <= MAX_LENGTH:
                filtered_arcs_for_percentile.append(arc)

    # Filter out circular annotation patterns (3+ arcs forming >200° circle)
    filtered_arcs_for_percentile = _filter_circular_annotation_patterns(
        filtered_arcs_for_percentile, page_width, page_height)

    # Hardcoded percentiles for door filtering
    door_min_percentile = 30
    door_max_percentile = 100

    # Calculate thresholds using filtered geometry only (no dust, no extremely long)
    line_strokes = [l['stroke_width'] for l in filtered_lines]
    arc_strokes = [a['stroke_width'] for a in filtered_arcs_for_percentile]
    all_strokes = line_strokes + arc_strokes

    if not all_strokes:
        return [], []

    min_threshold = np.percentile(all_strokes, door_min_percentile)
    max_threshold = np.percentile(all_strokes, door_max_percentile)

    # Calculate arc thresholds based on ARC stroke widths only
    if arc_strokes:
        arc_min_threshold = np.percentile(arc_strokes, 20)
        arc_max_threshold = np.percentile(arc_strokes, 100)
        if debug:
            print(
                f"DEBUG filter_door_candidates: Arc stroke width range: min={min(arc_strokes):.3f}, max={max(arc_strokes):.3f}, 20th={arc_min_threshold:.3f}, 100th={arc_max_threshold:.3f}")
    else:
        arc_min_threshold = arc_max_threshold = 0

    # Filter lines by stroke width
    door_lines = [line for line in filtered_lines if min_threshold <=
                  line['stroke_width'] <= max_threshold]

    # Filter arcs by stroke width
    door_arcs = [arc for arc in filtered_arcs_for_percentile if arc_min_threshold <=
                 arc['stroke_width'] <= arc_max_threshold]
    arcs_filtered_out = len(filtered_arcs_for_percentile) - len(door_arcs)

    if debug and arcs_filtered_out > 0:
        print(
            f"DEBUG filter_door_candidates: Filtered out {arcs_filtered_out} arcs, kept {len(door_arcs)} arcs")

    return door_lines, door_arcs


def analyze_geometry(lines: List[Dict], arcs: List[Dict], dashed_lines: List[Dict], page_width: float, page_height: float, debug: bool = False) -> Dict:
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
    if debug:
        print(f"\nDEBUG analyze_geometry: Initial extraction counts:")
        print(f"  Lines: {len(lines)}")
        print(f"  Arcs: {len(arcs)}")
        print(f"  Dashed lines: {len(dashed_lines)}")

    # Combine solid and dashed lines - door panels can be either
    all_lines = lines + dashed_lines

    # Step 1: Reconstruct arcs from tessellated line segments
    reconstructor = ArcReconstructor(page_width, page_height, debug=debug)
    reconstructed_arcs, used_line_indices, rejected_chains = reconstructor.reconstruct_arcs(
        all_lines)

    if reconstructed_arcs:
        if debug:
            print(
                f"DEBUG analyze_geometry: Reconstructed {len(reconstructed_arcs)} arcs from {len(used_line_indices)} tessellated segments")
        arcs = arcs + reconstructed_arcs
        # Optimized: use set for O(1) lookup instead of O(n) list check
        used_set = used_line_indices if isinstance(
            used_line_indices, set) else set(used_line_indices)
        all_lines = [line for i, line in enumerate(
            all_lines) if i not in used_set]
        if debug:
            print(
                f"DEBUG analyze_geometry: Removed {len(used_line_indices)} tessellated segments from lines list")

    # Step 2: Filter door candidates
    filtered_lines, filtered_arcs = filter_door_candidates(
        all_lines, arcs, page_width, page_height, debug=debug)

    if debug:
        print(
            f"DEBUG analyze_geometry: Number of filtered lines: {len(filtered_lines)}")
        print(
            f"DEBUG analyze_geometry: Number of filtered arcs: {len(filtered_arcs)}")

    # Step 3: Separate arcs for swing doors (< 120°) and double doors (150-210°)
    swing_door_arcs = []
    double_door_candidates = []

    for arc in filtered_arcs:
        sweep_angle = arc.get('sweep_angle')
        if sweep_angle is not None:
            if 150 <= sweep_angle <= 210:
                double_door_candidates.append(arc)
            elif sweep_angle < 120:  # Only include arcs < 120° for swing doors
                swing_door_arcs.append(arc)
        else:
            # If no sweep_angle, include in swing door arcs (will be filtered later)
            swing_door_arcs.append(arc)

    # Remove double door candidates from filtered_arcs so they're only drawn once (in BLUE, not RED)
    # Create a set of double door candidate IDs for efficient lookup
    double_door_ids = {id(arc) for arc in double_door_candidates}
    filtered_arcs_without_double_doors = [
        arc for arc in filtered_arcs if id(arc) not in double_door_ids]

    if debug:
        print(f"DEBUG analyze_geometry: Arc separation:")
        print(f"  Arcs < 120°: {len(swing_door_arcs)} (swing door candidates)")
        print(
            f"  Arcs 150-210°: {len(double_door_candidates)} (double door candidates)")
        if double_door_candidates:
            sweep_angles = [arc.get('sweep_angle')
                            for arc in double_door_candidates]
            print(
                f"  Double door candidate sweep angles: {[f'{s:.1f}°' for s in sweep_angles]}")

    return {
        "filtered_lines": filtered_lines,
        # Exclude double door candidates (they're drawn separately in BLUE)
        "filtered_arcs": filtered_arcs_without_double_doors,
        "door_candidate_arcs": swing_door_arcs,  # Only arcs < 120° for swing doors
        "dashed_lines": dashed_lines,
        "rejected_chains": rejected_chains,  # For visualization
        "double_door_candidates": double_door_candidates
    }
