"""Test to draw all arcs and lines, with reconstructed arcs in red."""
import sys
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_extractor import extract_vectors
from src.geometry_analyzer import analyze_geometry

# Path relative to project root
pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "FirstSource_R25-01360-A-V03.pdf_-_Page_2.pdf"

print("Extracting vectors...")
result = extract_vectors(str(pdf_path))

print(f"Extracted: {len(result['lines'])} lines, {len(result['arcs'])} arcs, {len(result['dashed_lines'])} dashed lines")

# Analyze geometry to get filtered lines and arcs
print("\nAnalyzing geometry...")
analysis = analyze_geometry(
    result['lines'],
    result['arcs'],
    result['dashed_lines'],
    result['page_width'],
    result['page_height']
)

filtered_lines = analysis['filtered_lines']
filtered_arcs = analysis['filtered_arcs']

# Separate reconstructed and original arcs
reconstructed_arcs = [arc for arc in filtered_arcs if arc.get('reconstructed', False)]
original_arcs = [arc for arc in filtered_arcs if not arc.get('reconstructed', False)]

print(f"\n=== SUMMARY ===")
print(f"Filtered lines: {len(filtered_lines)}")
print(f"Filtered arcs: {len(filtered_arcs)}")
print(f"  - Original arcs: {len(original_arcs)}")
print(f"  - Reconstructed arcs: {len(reconstructed_arcs)}")

# Create a new PDF document
new_doc = pymupdf.open()
new_page = new_doc.new_page(width=result['page_width'], height=result['page_height'])

# Draw filtered lines in black with accurate stroke widths
print("\nDrawing filtered lines...")
for line in filtered_lines:
    line_shape = new_page.new_shape()
    line_shape.draw_line(line['start'], line['end'])
    stroke_width = line.get('stroke_width', 1)
    is_dashed = line.get('is_dashed', False)
    if is_dashed:
        line_shape.finish(color=(0, 0, 0), width=stroke_width, dashes=[5, 5])
    else:
        line_shape.finish(color=(0, 0, 0), width=stroke_width)
    line_shape.commit()
print(f"  Completed: {len(filtered_lines)} lines drawn")

# Draw original arcs in black with accurate stroke widths
print("\nDrawing original arcs...")
for arc in original_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0 = tuple(control_points[0])
        p1 = tuple(control_points[1])
        p2 = tuple(control_points[2])
        p3 = tuple(control_points[3])
        
        arc_shape = new_page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        arc_shape.finish(color=(0, 0, 0), width=stroke_width)
        arc_shape.commit()
print(f"  Completed: {len(original_arcs)} original arcs drawn")

# Draw reconstructed arcs in RED with accurate stroke widths
print("\nDrawing reconstructed arcs in RED...")
for i, arc in enumerate(reconstructed_arcs):
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0 = tuple(control_points[0])
        p1 = tuple(control_points[1])
        p2 = tuple(control_points[2])
        p3 = tuple(control_points[3])
        
        arc_shape = new_page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        arc_shape.finish(color=(1, 0, 0), width=stroke_width)  # Red color, original stroke width
        arc_shape.commit()

print(f"  Completed: {len(reconstructed_arcs)} reconstructed arcs drawn in RED")

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_all_geometry.pdf"
new_doc.save(str(output_path))
new_doc.close()

print(f"\nOutput saved to: {output_path}")
