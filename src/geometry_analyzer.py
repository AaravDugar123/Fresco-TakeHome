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
        self.segment_min_threshold = page_diagonal * 0.0003
        self.gap_tolerance = page_diagonal * 0.0015

    def _is_short_segment(self, line: Dict) -> bool:
        """Check if line is short enough to be a tessellated segment (but not too small - dust)."""
        start = np.array(line['start'])
        end = np.array(line['end'])
        length = np.linalg.norm(end - start)
        return self.segment_min_threshold <= length < self.segment_max_threshold

    def _are_connected(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines are connected within tolerance."""
        start1 = np.array(line1['start'])
        end1 = np.array(line1['end'])
        start2 = np.array(line2['start'])
        end2 = np.array(line2['end'])

        distances = [
            np.linalg.norm(end1 - start2),
            np.linalg.norm(end1 - end2),
            np.linalg.norm(start1 - start2),
            np.linalg.norm(start1 - end2)
        ]
        return min(distances) <= self.gap_tolerance

    def _chain_segments(self, segments: List[Tuple[int, Dict]]) -> List[List[Tuple[int, Dict]]]:
        """Group connected segments into chains."""
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

                    if connects_to_end:
                        chain.append((other_orig_idx, other_seg))
                        used.add(j)
                        changed = True
                        break
                    elif connects_to_start:
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

    def _fit_circle(self, chain: List[Tuple[int, Dict]]) -> Optional[Dict]:
        """Fit circle to chain of segments using geometric circle fitting."""
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

        if len(points) < 3:
            return None

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
                return None

            t_s = np.linalg.solve(A, b)
            center = mid1 + t_s[0] * perp1
            radius = np.linalg.norm(p0 - center)

            max_error = 0
            for p in points_array:
                dist = np.abs(np.linalg.norm(p - center) - radius)
                max_error = max(max_error, dist)

            if max_error > radius * 0.8:
                return None

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

            # Ensure arc is open (not closed)
            if sweep_angle_deg >= 360 or sweep_angle_deg < 15:
                return None
            if sweep_angle_deg > 200:
                return None

            chord_length = np.linalg.norm(p_end - p0)
            if chord_length < 1e-5:
                return None

            arc_length = radius * sweep_angle
            if arc_length <= chord_length * 1.004:
                return None

            page_diagonal = np.sqrt(self.page_width**2 + self.page_height**2)
            min_radius = page_diagonal * 0.00125
            max_radius = page_diagonal * 0.15
            if radius < min_radius or radius > max_radius:
                return None

            chord_radius_ratio = chord_length / radius if radius > 0 else 0
            if chord_radius_ratio < 0.4 or chord_radius_ratio > 3.6:
                return None

            # Calculate tangent directions
            start_tangent = np.array(
                [-np.sin(start_angle), np.cos(start_angle)])
            end_tangent = np.array([-np.sin(end_angle), np.cos(end_angle)])

            if diff < 0:
                start_tangent = -start_tangent
                end_tangent = -end_tangent

            sweep_rad = sweep_angle
            if sweep_rad < 1e-5:
                return None

            control_distance = radius * (4.0 / 3.0) * np.tan(sweep_rad / 4.0)

            p1 = p0 + control_distance * start_tangent
            p2 = p_end - control_distance * end_tangent

            if np.linalg.norm(p1 - p0) < 1e-5 or np.linalg.norm(p2 - p_end) < 1e-5:
                return None
            if np.linalg.norm(p1 - p2) < 1e-5:
                return None

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
            }
        except Exception:
            return None

    def reconstruct_arcs(self, lines: List[Dict]) -> Tuple[List[Dict], set]:
        """Reconstruct arcs from tessellated line segments."""
        start_time = time.time()

        segments_with_indices = []
        for i, line in enumerate(lines):
            if self._is_short_segment(line):
                segments_with_indices.append((i, line))

        segment_time = time.time()
        print(
            f"DEBUG ArcReconstructor: Found {len(segments_with_indices)} candidate segments out of {len(lines)} total lines (took {segment_time - start_time:.3f}s)")

        if len(segments_with_indices) < 3:
            return [], set()

        chains = self._chain_segments(segments_with_indices)
        chain_time = time.time()
        print(
            f"DEBUG ArcReconstructor: Formed {len(chains)} chains from segments (took {chain_time - segment_time:.3f}s)")

        if not chains:
            return [], set()

        reconstructed_arcs = []
        used_line_indices = set()
        rejected_detour = 0
        rejected_fit = 0

        for chain_idx, chain in enumerate(chains):
            detour_index = self._calculate_detour_index(chain)
            if detour_index > 1.02:
                arc = self._fit_circle(chain)
                if arc is not None:
                    reconstructed_arcs.append(arc)
                    for orig_idx, _ in chain:
                        used_line_indices.add(orig_idx)
                    print(
                        f"DEBUG ArcReconstructor: Chain {chain_idx}: Reconstructed arc (radius={arc['radius']:.2f}, sweep={arc['sweep_angle']:.1f}°)")
                else:
                    rejected_fit += 1
            else:
                rejected_detour += 1

        fit_time = time.time()
        total_time = fit_time - start_time

        print(
            f"DEBUG ArcReconstructor: Rejected {rejected_detour} chains (detour), {rejected_fit} chains (fit) (fitting took {fit_time - chain_time:.3f}s)")
        print(
            f"DEBUG ArcReconstructor: Final result: {len(reconstructed_arcs)} reconstructed arcs from {len(used_line_indices)} line segments")
        print(
            f"DEBUG ArcReconstructor: Total time taken: {total_time:.3f} seconds")
        return reconstructed_arcs, used_line_indices


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
        arc_min_threshold = np.percentile(arc_strokes, 40)
        arc_max_threshold = np.percentile(arc_strokes, 80)
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
    reconstructed_arcs, used_line_indices = reconstructor.reconstruct_arcs(
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
        "dashed_lines": dashed_lines
    }
