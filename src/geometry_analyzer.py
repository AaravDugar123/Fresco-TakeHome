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

    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height
        page_diagonal = np.sqrt(page_width**2 + page_height**2)
        self.segment_max_threshold = page_diagonal * 0.003
        self.segment_min_threshold = page_diagonal * 0.00035
        self.gap_tolerance = page_diagonal * 0.0016

    def _is_short_segment(self, line: Dict) -> bool:
        """Check if line is short enough to be a tessellated segment (but not too small - dust)."""
        start = line['start']
        end = line['end']
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx*dx + dy*dy) ** 0.5
        return self.segment_min_threshold <= length < self.segment_max_threshold

    def _are_connected(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines are connected within tolerance."""
        end1 = line1['end']
        start1 = line1['start']
        start2 = line2['start']
        end2 = line2['end']

        threshold_sq = self.gap_tolerance * self.gap_tolerance

        # Check all 4 endpoint combinations with squared distances (avoid sqrt)
        dx = end1[0] - start2[0]
        dy = end1[1] - start2[1]
        if dx*dx + dy*dy <= threshold_sq:
            return True

        dx = end1[0] - end2[0]
        dy = end1[1] - end2[1]
        if dx*dx + dy*dy <= threshold_sq:
            return True

        dx = start1[0] - start2[0]
        dy = start1[1] - start2[1]
        if dx*dx + dy*dy <= threshold_sq:
            return True

        dx = start1[0] - end2[0]
        dy = start1[1] - end2[1]
        return dx*dx + dy*dy <= threshold_sq

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
            dist_to_start = np.linalg.norm(last_end - new_start)
            dist_to_end = np.linalg.norm(last_end - new_end)

            if dist_to_start <= dist_to_end:
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
            dist_to_start = np.linalg.norm(first_start - new_start)
            dist_to_end = np.linalg.norm(first_start - new_end)

            if dist_to_start <= dist_to_end:
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

    def _chain_segments(self, segments: List[Tuple[int, Dict]]) -> List[List[Tuple[int, Dict]]]:
        """Group connected segments into chains, avoiding intersecting lines."""
        if not segments:
            return []

        chains = []
        used = set()

        for i, (orig_idx, seg) in enumerate(segments):
            if i in used:
                continue

            chain = [(orig_idx, seg)]
            used.add(i)
            changed = True

            while changed:
                changed = False
                for j, (other_orig_idx, other_seg) in enumerate(segments):
                    if j in used:
                        continue

                    connects_to_end = self._are_connected(
                        chain[-1][1], other_seg)
                    connects_to_start = self._are_connected(
                        chain[0][1], other_seg)

                    # Check if segment continues smoothly before adding
                    if connects_to_end:
                        if self._continues_smoothly(chain, other_seg, add_to_end=True):
                            chain.append((other_orig_idx, other_seg))
                            used.add(j)
                            changed = True
                            break
                    elif connects_to_start:
                        if self._continues_smoothly(chain, other_seg, add_to_end=False):
                            chain.insert(0, (other_orig_idx, other_seg))
                            used.add(j)
                            changed = True
                            break

            if len(chain) >= 3:
                chains.append(chain)

        return chains

    def _calculate_detour_index(self, chain: List[Tuple[int, Dict]]) -> float:
        """Calculate detour index: Total Path Length / Straight Distance."""
        if len(chain) < 2:
            return 1.0

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
                dist_to_start = np.linalg.norm(start - current_point)
                dist_to_end = np.linalg.norm(end - current_point)

                if dist_to_start <= dist_to_end and dist_to_start <= self.gap_tolerance:
                    points.append(end)
                    current_point = end
                elif dist_to_end <= self.gap_tolerance:
                    points.append(start)
                    current_point = start
                else:
                    points.append(start)
                    points.append(end)
                    current_point = end

        if len(points) < 2:
            return 1.0

        total_length = 0.0
        for i in range(len(points) - 1):
            total_length += np.linalg.norm(points[i+1] - points[i])

        straight_distance = np.linalg.norm(points[-1] - points[0])
        if straight_distance < 1e-5:
            return 1.0

        return total_length / straight_distance

    def _fit_circle(self, chain: List[Tuple[int, Dict]]) -> Tuple[Optional[Dict], Optional[str], Optional[Dict]]:
        """Fit circle to chain of segments using geometric circle fitting.
        Returns (arc_dict, diagnostic_string, metrics_dict) where:
        - arc_dict is the arc on success, None on failure
        - diagnostic_string is None on success, or contains the rejection reason on failure
        - metrics_dict contains computed values (radius, sweep_angle, etc.) for debugging"""
        points = []
        current_point = None

        for orig_idx, line in chain:
            start = line['start']
            end = line['end']
            start_arr = np.array(start)
            end_arr = np.array(end)

            if len(points) == 0:
                points.append(start_arr)
                points.append(end_arr)
                current_point = end
            else:
                current_arr = np.array(current_point)
                dist_to_start = np.linalg.norm(start_arr - current_arr)
                dist_to_end = np.linalg.norm(end_arr - current_arr)

                if dist_to_start <= dist_to_end and dist_to_start <= self.gap_tolerance:
                    points.append(end_arr)
                    current_point = end
                elif dist_to_end <= self.gap_tolerance:
                    points.append(start_arr)
                    current_point = start
                else:
                    points.append(start_arr)
                    points.append(end_arr)
                    current_point = end

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

            # Store metrics for debugging
            metrics = {'radius': radius, 'max_error': max_error,
                       'error_threshold': radius * .6}

            if max_error > radius * .6:
                return None, f"max_error_too_high({max_error:.2f} > {radius * .6:.2f}, radius={radius:.2f}, margin={max_error - radius * 0.6:.2f})", metrics

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

            # Update metrics
            metrics['sweep_angle_deg'] = sweep_angle_deg

            # Ensure arc is open (not closed) - stricter range
            if sweep_angle_deg >= 360 or sweep_angle_deg < 12:
                if sweep_angle_deg < 12:
                    margin = sweep_angle_deg - 12
                    return None, f"sweep_angle_out_of_range({sweep_angle_deg:.1f}° < 12°, margin={margin:.1f}°)", metrics
                else:
                    return None, f"sweep_angle_out_of_range({sweep_angle_deg:.1f}° >= 360°)", metrics
            if sweep_angle_deg > 120:
                return None, f"sweep_angle_too_large({sweep_angle_deg:.1f}° > 120°, margin={sweep_angle_deg - 200:.1f}°)", metrics

            chord_length = np.linalg.norm(p_end - p0)
            if chord_length < 1e-1:
                return None, "chord_length_too_small", metrics

            arc_length = radius * sweep_angle
            metrics['chord_length'] = chord_length
            metrics['arc_length'] = arc_length
            metrics['arc_chord_ratio'] = arc_length / \
                chord_length if chord_length > 0 else 0

            if arc_length <= chord_length * 1.01:
                return None, f"arc_too_flat(arc={arc_length:.2f}, chord={chord_length:.2f}, ratio={arc_length/chord_length:.4f}, need >1.005, margin={arc_length/chord_length - 1.005:.4f})", metrics

            page_diagonal = np.sqrt(self.page_width**2 + self.page_height**2)
            min_radius = page_diagonal * 0.0015  # Stricter: increased from 0.00125
            max_radius = page_diagonal * 0.1  # Stricter: reduced from 0.15
            metrics['page_diagonal'] = page_diagonal
            metrics['min_radius'] = min_radius
            metrics['max_radius'] = max_radius

            if radius < min_radius or radius > max_radius:
                margin = min_radius - radius if radius < min_radius else radius - max_radius
                return None, f"radius_out_of_range({radius:.2f}, valid: {min_radius:.2f}-{max_radius:.2f}, page_diag={page_diagonal:.2f}, margin={margin:.2f})", metrics

            chord_radius_ratio = chord_length / radius if radius > 0 else 0
            metrics['chord_radius_ratio'] = chord_radius_ratio

            if chord_radius_ratio < 0.3 or chord_radius_ratio > 3.2:
                margin = 0.3 - chord_radius_ratio if chord_radius_ratio < 0.3 else chord_radius_ratio - 3.2
                return None, f"chord_radius_ratio_out_of_range({chord_radius_ratio:.2f}, valid: 0.3-3.2, margin={margin:.2f})", metrics

            # Calculate tangent directions
            start_tangent = np.array(
                [-np.sin(start_angle), np.cos(start_angle)])
            end_tangent = np.array([-np.sin(end_angle), np.cos(end_angle)])

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

            control_points_list = [
                [float(p0[0]), float(p0[1])],
                [float(p1[0]), float(p1[1])],
                [float(p2[0]), float(p2[1])],
                [float(p_end[0]), float(p_end[1])]
            ]

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
        start_time = time.time()

        segments_with_indices = []
        for i, line in enumerate(lines):
            if self._is_short_segment(line):
                segments_with_indices.append((i, line))

        segment_time = time.time()
        print(
            f"DEBUG ArcReconstructor: Found {len(segments_with_indices)} candidate segments out of {len(lines)} total lines (took {segment_time - start_time:.3f}s)")

        if len(segments_with_indices) < 3:
            return [], set(), []

        chains = self._chain_segments(segments_with_indices)
        chain_time = time.time()
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

        for chain_idx, chain in enumerate(chains):
            # Calculate chain bounding box for location identification
            all_x = []
            all_y = []
            for orig_idx, line in chain:
                all_x.extend([line['start'][0], line['end'][0]])
                all_y.extend([line['start'][1], line['end'][1]])
            chain_bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
            chain_center = ((min(all_x) + max(all_x)) / 2,
                            (min(all_y) + max(all_y)) / 2)

            detour_index = self._calculate_detour_index(chain)
            if detour_index > 1.01:  # Stricter: increased from 1.01
                arc, diagnostic, metrics = self._fit_circle(chain)
                if arc is not None:
                    reconstructed_arcs.append(arc)
                    for orig_idx, _ in chain:
                        used_line_indices.add(orig_idx)
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
                    print(
                        f"DEBUG ArcReconstructor: Chain {chain_idx}: Rejected fit - detour={detour_index:.4f}, reason={diagnostic}, center=({chain_center[0]:.1f}, {chain_center[1]:.1f}), bbox=({chain_bbox[0]:.1f},{chain_bbox[1]:.1f})-({chain_bbox[2]:.1f},{chain_bbox[3]:.1f})")

                    # TEMPORARY: Log detailed info for chains near target area (894.4, 1587)
                    target_x, target_y = 894.4, 1587.0
                    tolerance = 50.0  # Within 50 units
                    if abs(chain_center[0] - target_x) < tolerance and abs(chain_center[1] - target_y) < tolerance:
                        print(
                            f"\n*** TARGET AREA REJECTION (near {target_x}, {target_y}) ***")
                        print(
                            f"  Chain {chain_idx}: center=({chain_center[0]:.1f}, {chain_center[1]:.1f})")
                        print(f"  Rejection reason: {diagnostic}")
                        print(f"  Detour index: {detour_index:.4f}")
                        print(
                            f"  Bbox: ({chain_bbox[0]:.1f}, {chain_bbox[1]:.1f}) to ({chain_bbox[2]:.1f}, {chain_bbox[3]:.1f})")
                        if metrics:
                            print(f"  Metrics:")
                            if metrics.get('radius') is not None:
                                print(f"    radius: {metrics['radius']:.2f}")
                            if metrics.get('sweep_angle_deg') is not None:
                                print(
                                    f"    sweep_angle: {metrics['sweep_angle_deg']:.1f}°")
                            if metrics.get('chord_radius_ratio') is not None:
                                print(
                                    f"    chord_radius_ratio: {metrics['chord_radius_ratio']:.2f}")
                            if metrics.get('error_threshold') is not None:
                                print(
                                    f"    error_threshold: {metrics['error_threshold']:.2f}")
                            if metrics.get('max_error') is not None:
                                print(
                                    f"    max_error: {metrics['max_error']:.2f}")
                        print(f"*** END TARGET AREA REJECTION ***\n")
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
                print(
                    f"DEBUG ArcReconstructor: Chain {chain_idx}: Rejected detour - detour_index={detour_index:.4f} (need >1.015), center=({chain_center[0]:.1f}, {chain_center[1]:.1f})")

                # TEMPORARY: Log detailed info for chains near target area (894.4, 1587)
                target_x, target_y = 894.4, 1587.0
                tolerance = 50.0  # Within 50 units
                if abs(chain_center[0] - target_x) < tolerance and abs(chain_center[1] - target_y) < tolerance:
                    print(
                        f"\n*** TARGET AREA REJECTION (near {target_x}, {target_y}) ***")
                    print(
                        f"  Chain {chain_idx}: center=({chain_center[0]:.1f}, {chain_center[1]:.1f})")
                    print(
                        f"  Rejection reason: detour_too_low (detour_index={detour_index:.4f} <= 1.015)")
                    print(f"  Detour index: {detour_index:.4f}")
                    print(
                        f"  Bbox: ({chain_bbox[0]:.1f}, {chain_bbox[1]:.1f}) to ({chain_bbox[2]:.1f}, {chain_bbox[3]:.1f})")
                    print(f"*** END TARGET AREA REJECTION ***\n")

        fit_time = time.time()
        total_time = fit_time - start_time

        print(
            f"DEBUG ArcReconstructor: Rejected {rejected_detour} chains (detour), {rejected_fit} chains (fit) (fitting took {fit_time - chain_time:.3f}s)")

        if fit_rejection_details:
            print(f"\nDEBUG ArcReconstructor: Fit rejection summary:")
            rejection_reasons = {}
            for chain_idx, detour, reason, metrics, center, bbox in fit_rejection_details:
                reason_type = reason.split('(')[0] if '(' in reason else reason
                rejection_reasons[reason_type] = rejection_reasons.get(
                    reason_type, 0) + 1
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} chains")

            print(f"\nDEBUG ArcReconstructor: Detailed rejection info (first 20 chains):")
            for chain_idx, detour, reason, metrics, center, bbox in fit_rejection_details[:20]:
                print(f"  Chain {chain_idx}:")
                print(
                    f"    Location: center=({center[0]:.1f}, {center[1]:.1f}), bbox=({bbox[0]:.1f},{bbox[1]:.1f})-({bbox[2]:.1f},{bbox[3]:.1f})")
                print(f"    Detour index: {detour:.4f}")
                print(f"    Rejection reason: {reason}")
                if metrics:
                    radius_val = metrics.get('radius')
                    sweep_val = metrics.get('sweep_angle_deg')
                    ratio_val = metrics.get('chord_radius_ratio')
                    radius_str = f"{radius_val:.2f}" if radius_val is not None else "N/A"
                    sweep_str = f"{sweep_val:.1f}°" if sweep_val is not None else "N/A"
                    ratio_str = f"{ratio_val:.2f}" if ratio_val is not None else "N/A"
                    print(
                        f"    Metrics: radius={radius_str}, sweep_angle={sweep_str}, chord_radius_ratio={ratio_str}")

        print(
            f"\nDEBUG ArcReconstructor: Final result: {len(reconstructed_arcs)} reconstructed arcs from {len(used_line_indices)} line segments")
        print(
            f"DEBUG ArcReconstructor: Total time taken: {total_time:.3f} seconds")
        return reconstructed_arcs, used_line_indices, rejected_chains


def _filter_circular_annotation_patterns(arcs: List[Dict], page_width: float, page_height: float) -> List[Dict]:
    """
    Filter out groups of arcs that form circular annotation patterns (3+ arcs forming >200° circle).
    Optimized: builds connectivity graph once, uses BFS to find all connected components.
    """
    if len(arcs) < 3:
        return arcs  # Need at least 3 arcs to form a circle pattern
    
    page_diagonal = np.sqrt(page_width**2 + page_height**2)
    touch_tolerance_sq = (page_diagonal * 0.002) ** 2  # Squared distance for speed
    
    n = len(arcs)
    
    # Pre-compute endpoints and sweep angles (cache for speed)
    arc_data = []
    for arc in arcs:
        cp = arc.get('control_points')
        if not cp or len(cp) != 4:
            arc_data.append(None)
            continue
        
        start = np.array(cp[0])
        end = np.array(cp[3])
        
        # Get sweep angle (cache it)
        sweep = arc.get('sweep_angle')
        if sweep is None:
            result = get_bezier_radius(cp)
            if result is None:
                arc_data.append(None)
                continue
            radius, _ = result
            from src.door_classifier import calculate_arc_sweep_angle
            sweep = calculate_arc_sweep_angle(arc, radius)
            if sweep is None:
                arc_data.append(None)
                continue
        
        arc_data.append({
            'start': start,
            'end': end,
            'sweep': sweep
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
        
        # Check if this component matches the pattern (3+ arcs, >200° total)
        if len(component) >= 3:
            total_sweep = sum(arc_data[idx]['sweep'] for idx in component if arc_data[idx] is not None)
            if total_sweep > 200:
                arc_indices_to_remove.update(component)
    
    if arc_indices_to_remove:
        filtered_arcs = [arc for idx, arc in enumerate(arcs) if idx not in arc_indices_to_remove]
        print(f"DEBUG filter_door_candidates: Filtered out {len(arc_indices_to_remove)} arcs forming circular annotation patterns")
        return filtered_arcs
    
    return arcs


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
    MIN_LENGTH = page_diagonal * 0.005  # 0.5% of page diagonal (dust)
    MAX_LENGTH = page_diagonal * 0.05  # 6% of page diagonal (extremely long)

    # Filter out dust and extremely long lines
    filtered_lines = []
    for line in lines:
        start = line['start']
        end = line['end']
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx*dx + dy*dy) ** 0.5
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
    
    # Filter out circular annotation patterns (3+ arcs forming >200° circle)
    filtered_arcs_for_percentile = _filter_circular_annotation_patterns(
        filtered_arcs_for_percentile, page_width, page_height)

    # Hardcoded percentiles for door filtering
    door_min_percentile = 40
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
        arc_max_threshold = np.percentile(arc_strokes, 100)
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

    # Step 1: Reconstruct arcs from tessellated line segments
    reconstructor = ArcReconstructor(page_width, page_height)
    reconstructed_arcs, used_line_indices, rejected_chains = reconstructor.reconstruct_arcs(
        all_lines)

    if reconstructed_arcs:
        print(
            f"DEBUG analyze_geometry: Reconstructed {len(reconstructed_arcs)} arcs from {len(used_line_indices)} tessellated segments")
        arcs = arcs + reconstructed_arcs
        all_lines = [line for i, line in enumerate(
            all_lines) if i not in used_line_indices]
        print(
            f"DEBUG analyze_geometry: Removed {len(used_line_indices)} tessellated segments from lines list")

    # Step 2: Filter door candidates
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
        "dashed_lines": dashed_lines,
        "rejected_chains": rejected_chains  # For visualization
    }
