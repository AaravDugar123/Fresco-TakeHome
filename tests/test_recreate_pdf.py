"""Test to draw all arcs in red over the original PDF."""
from src.geometry_analyzer import analyze_geometry, get_bezier_radius
from src.vector_extractor import extract_vectors
import sys
from pathlib import Path
import pymupdf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "CopellIndependent_NA02-01_-_FLOOR_PLAN_-LEVEL_ONE_-_CoppellIndependent.pdf_-_Page_30.pdf"

print("Extracting vectors...")
result = extract_vectors(str(pdf_path))

print(
    f"Extracted: {len(result['lines'])} lines, {len(result['arcs'])} arcs, {len(result['dashed_lines'])} dashed lines")

print("\nAnalyzing geometry...")
analysis = analyze_geometry(
    result['lines'],
    result['arcs'],
    result['dashed_lines'],
    result['page_width'],
    result['page_height'],
    debug=False  
)

filtered_arcs = analysis['filtered_arcs']
double_door_candidates = analysis.get('double_door_candidates', [])
print(f"\nDrawing {len(filtered_arcs)} arcs in red over original...")
print(
    f"Drawing {len(double_door_candidates)} double door candidate arcs (150-210°) in blue...")

doc = pymupdf.open(str(pdf_path))
page = doc[0]

# Draw arcs in red
centers_drawn = 0
for arc in filtered_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0, p1, p2, p3 = tuple(control_points[0]), tuple(
            control_points[1]), tuple(control_points[2]), tuple(control_points[3])

        arc_shape = page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        visible_width = max(stroke_width, 2.0)
        arc_shape.finish(color=(1, 0, 0), width=visible_width)
        arc_shape.commit()

        if 'center' in arc and 'radius' in arc:
            center = tuple(arc['center'])
        else:
            result = get_bezier_radius(control_points)
            if result is not None:
                radius, arc_center = result
                center = tuple(arc_center)
            else:
                center = None

        if center is not None:
            center_x = int(center[0])
            center_y = int(center[1])
            coord_text = f"({center_x}, {center_y})"

            point = pymupdf.Point(center[0], center[1])
            page.insert_text(
                point,
                coord_text,
                fontsize=8,
                color=(0, 0, 1),  
                render_mode=0
            )
            centers_drawn += 1

# Draw douuble door canidates
double_door_centers_drawn = 0
for arc in double_door_candidates:
    control_points = arc.get('control_points')
    if control_points and len(control_points) == 4:
        p0, p1, p2, p3 = tuple(control_points[0]), tuple(
            control_points[1]), tuple(control_points[2]), tuple(control_points[3])

        # Draw arcs
        arc_shape = page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        visible_width = max(stroke_width, 2.0)
        arc_shape.finish(color=(0, 0, 1), width=visible_width)  # Blue color
        arc_shape.commit()

        if 'center' in arc and 'radius' in arc:
            center = tuple(arc['center'])
        else:
            result = get_bezier_radius(control_points)
            if result is not None:
                radius, arc_center = result
                center = tuple(arc_center)
            else:
                center = None

        # Draw cords
        if center is not None:
            center_x = int(center[0])
            center_y = int(center[1])
            coord_text = f"({center_x}, {center_y})"

            point = pymupdf.Point(center[0], center[1])
            page.insert_text(
                point,
                coord_text,
                fontsize=8,
                color=(0, 0, 1),  
                render_mode=0  
            )
            double_door_centers_drawn += 1

print(
    f"  Completed: {len(filtered_arcs)} arcs drawn in RED, {centers_drawn} center coordinates drawn")
print(
    f"  Completed: {len(double_door_candidates)} double door candidate arcs drawn in BLUE, {double_door_centers_drawn} center coordinates drawn")


output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_arcs_overlay.pdf"
doc.save(str(output_path))
doc.close()

print(f"\nOutput saved to: {output_path}")
