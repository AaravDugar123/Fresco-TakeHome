"""Test to draw all arcs in red over the original PDF."""
from src.geometry_analyzer import analyze_geometry, get_bezier_radius
from src.vector_extractor import extract_vectors
import sys
from pathlib import Path
import pymupdf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# Path relative to project root
pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "FirstSource_R25-01360-A-V03.pdf_-_Page_2.pdf"

print("Extracting vectors...")
result = extract_vectors(str(pdf_path))

print(
    f"Extracted: {len(result['lines'])} lines, {len(result['arcs'])} arcs, {len(result['dashed_lines'])} dashed lines")

# Analyze geometry to get filtered lines and arcs
print("\nAnalyzing geometry...")
analysis = analyze_geometry(
    result['lines'],
    result['arcs'],
    result['dashed_lines'],
    result['page_width'],
    result['page_height'],
    debug=False  # Disable debug prints for faster execution
)

filtered_arcs = analysis['filtered_arcs']
print(f"\nDrawing {len(filtered_arcs)} arcs in RED over original...")

# Open the original PDF
doc = pymupdf.open(str(pdf_path))
page = doc[0]

# Draw all arcs in RED over the original and their centers
centers_drawn = 0
for arc in filtered_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0, p1, p2, p3 = tuple(control_points[0]), tuple(
            control_points[1]), tuple(control_points[2]), tuple(control_points[3])

        # Draw the arc in RED
        arc_shape = page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        visible_width = max(stroke_width, 2.0)
        arc_shape.finish(color=(1, 0, 0), width=visible_width)
        arc_shape.commit()

        # Calculate center using the same method as classifier
        # The classifier uses get_bezier_radius() which returns (radius, midpoint)
        # where midpoint is the point on the curve at t=0.5, not the actual circle center
        # This matches what the classifier debug output shows
        result = get_bezier_radius(control_points)
        if result is not None:
            radius, arc_center = result
            # Use the midpoint (arc_center) which is what classifier shows
            center = tuple(arc_center)
        else:
            # If get_bezier_radius fails, try using stored center if available
            if 'center' in arc:
                center = tuple(arc['center'])
            else:
                center = None

        # Draw center coordinates as text
        if center is not None:
            # Round coordinates to integers for matching with classifier output
            center_x = round(center[0])
            center_y = round(center[1])
            coord_text = f"({center_x}, {center_y})"

            # Insert text at the center point
            point = pymupdf.Point(center[0], center[1])
            page.insert_text(
                point,
                coord_text,
                fontsize=8,
                color=(0, 0, 1),  # Blue text
                render_mode=0  # Fill text
            )
            centers_drawn += 1

print(
    f"  Completed: {len(filtered_arcs)} arcs drawn in RED, {centers_drawn} center coordinates drawn in BLUE")

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_arcs_overlay.pdf"
doc.save(str(output_path))
doc.close()

print(f"\nOutput saved to: {output_path}")
