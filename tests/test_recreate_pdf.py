"""Test to draw all arcs in red over the original PDF."""
from src.geometry_analyzer import analyze_geometry, get_bezier_radius
from src.vector_extractor import extract_vectors
import sys
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).parent.parent))


# Path relative to project root
pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "AC_Convention_Center_ac_cc.pdf_-_Page_122.pdf"

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
    result['page_height']
)

filtered_arcs = analysis['filtered_arcs']

# Separate reconstructed and original arcs
reconstructed_arcs = [
    arc for arc in filtered_arcs if arc.get('reconstructed', False)]
original_arcs = [arc for arc in filtered_arcs if not arc.get(
    'reconstructed', False)]

print(f"\n=== SUMMARY ===")
print(f"Filtered arcs: {len(filtered_arcs)}")
print(f"  - Original arcs: {len(original_arcs)}")
print(f"  - Reconstructed arcs: {len(reconstructed_arcs)}")

# Open the original PDF
print("\nOpening original PDF...")
doc = pymupdf.open(str(pdf_path))
page = doc[0]

# Draw all arcs in RED over the original
print(f"\nDrawing {len(filtered_arcs)} arcs in RED over original...")
for arc in filtered_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0 = tuple(control_points[0])
        p1 = tuple(control_points[1])
        p2 = tuple(control_points[2])
        p3 = tuple(control_points[3])

        arc_shape = page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        # Ensure minimum stroke width for visibility
        visible_width = max(stroke_width, 2.0)
        # Red color for all arcs
        arc_shape.finish(color=(1, 0, 0), width=visible_width)
        arc_shape.commit()
        
        # Draw center coordinates of the arc
        center_result = get_bezier_radius(control_points)
        if center_result:
            arc_radius, arc_center = center_result
            center_x = float(arc_center[0])
            center_y = float(arc_center[1])
            
            # Draw coordinate text at the center point
            coord_text = f"({center_x:.0f}, {center_y:.0f})"
            page.insert_text(
                pymupdf.Point(center_x, center_y),
                coord_text,
                fontsize=8,
                color=(1, 0, 0)  # Red text
            )

print(f"  Completed: {len(filtered_arcs)} arcs drawn in RED with center points")

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_arcs_overlay.pdf"
doc.save(str(output_path))
doc.close()

print(f"\nOutput saved to: {output_path}")
