"""Test to draw all arcs in red over the original PDF."""
from src.geometry_analyzer import analyze_geometry
from src.vector_extractor import extract_vectors
import sys
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).parent.parent))


# Path relative to project root
pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "CopellIndependent_NA02-01_-_FLOOR_PLAN_-LEVEL_ONE_-_CoppellIndependent.pdf_-_Page_30.pdf"

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
print(f"\nDrawing {len(filtered_arcs)} arcs in RED over original...")

# Open the original PDF
doc = pymupdf.open(str(pdf_path))
page = doc[0]

# Draw all arcs in RED over the original (no expensive center calculations)
for arc in filtered_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0, p1, p2, p3 = tuple(control_points[0]), tuple(control_points[1]), tuple(control_points[2]), tuple(control_points[3])
        
        arc_shape = page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        visible_width = max(stroke_width, 2.0)
        arc_shape.finish(color=(1, 0, 0), width=visible_width)
        arc_shape.commit()

print(f"  Completed: {len(filtered_arcs)} arcs drawn in RED")

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_arcs_overlay.pdf"
doc.save(str(output_path))
doc.close()

print(f"\nOutput saved to: {output_path}")
