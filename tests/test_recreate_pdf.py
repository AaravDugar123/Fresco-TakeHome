"""Test to recreate PDF using only extracted lines and curves."""
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

print(f"\nAfter filtering:")
print(f"  Filtered lines: {len(filtered_lines)}")
print(f"  Filtered arcs: {len(filtered_arcs)}")

# Create a new PDF document
new_doc = pymupdf.open()  # Create empty document
new_page = new_doc.new_page(width=result['page_width'], height=result['page_height'])

# Draw filtered lines (door candidates)
print("\nDrawing filtered lines (door candidates)...")
for line in filtered_lines:
    start = line['start']
    end = line['end']
    stroke_width = line['stroke_width']
    is_dashed = line.get('is_dashed', False)
    if is_dashed:
        new_page.draw_line(start, end, color=(0, 0, 0), width=stroke_width, dashes=[5, 5])
    else:
        new_page.draw_line(start, end, color=(0, 0, 0), width=stroke_width)

# Draw filtered arcs (door candidates, circles already filtered out)
print("Drawing filtered arcs (door candidates)...")
for arc in filtered_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0, p1, p2, p3 = control_points
        stroke_width = arc['stroke_width']
        # Draw cubic Bezier curve
        new_page.draw_bezier(p0, p1, p2, p3, color=(0, 0, 0), width=stroke_width)

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_recreated.pdf"
new_doc.save(str(output_path))
new_doc.close()

print(f"\nRecreated PDF saved to: {output_path}")
print(f"Page size: {result['page_width']} x {result['page_height']}")
