"""Test to draw all arcs and lines, with reconstructed arcs in red."""
from src.geometry_analyzer import analyze_geometry
from src.vector_extractor import extract_vectors
import sys
from pathlib import Path
import pymupdf

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
    result['page_height']
)

filtered_lines = analysis['filtered_lines']
filtered_arcs = analysis['filtered_arcs']

# Separate reconstructed and original arcs
reconstructed_arcs = [
    arc for arc in filtered_arcs if arc.get('reconstructed', False)]
original_arcs = [arc for arc in filtered_arcs if not arc.get(
    'reconstructed', False)]

print(f"\n=== SUMMARY ===")
print(f"Filtered lines: {len(filtered_lines)}")
print(f"Filtered arcs: {len(filtered_arcs)}")
print(f"  - Original arcs: {len(original_arcs)}")
print(f"  - Reconstructed arcs: {len(reconstructed_arcs)}")

# Create a new PDF document
new_doc = pymupdf.open()
new_page = new_doc.new_page(
    width=result['page_width'], height=result['page_height'])

# Add coordinate labels on bottom and left edges (no grid lines)
print("\nAdding coordinate labels...")
page_width = result['page_width']
page_height = result['page_height']
label_spacing = 100  # Labels every 100 units
font_size = 8
label_offset = 10  # Offset to avoid cutoff

# X-axis labels (bottom edge) - PDF origin is bottom-left
for x in range(0, int(page_width) + label_spacing, label_spacing):
    if x <= page_width - 20:  # Ensure label fits within page
        new_page.insert_text(
            pymupdf.Point(x, label_offset),
            str(x),
            fontsize=font_size,
            color=(0.5, 0.5, 0.5)
        )

# Y-axis labels (left edge) - PDF origin is bottom-left
for y in range(0, int(page_height) + label_spacing, label_spacing):
    if y <= page_height - 10:  # Ensure label fits within page
        new_page.insert_text(
            pymupdf.Point(label_offset, y),
            str(y),
            fontsize=font_size,
            color=(0.5, 0.5, 0.5)
        )

print(f"  Coordinate labels added: {label_spacing} unit spacing")

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

# Draw all arcs in RED (both original and reconstructed) with coordinates
print("\nDrawing all arcs in RED with coordinates...")
from src.geometry_analyzer import get_bezier_radius

for arc in filtered_arcs:
    control_points = arc['control_points']
    if len(control_points) == 4:
        p0 = tuple(control_points[0])
        p1 = tuple(control_points[1])
        p2 = tuple(control_points[2])
        p3 = tuple(control_points[3])

        arc_shape = new_page.new_shape()
        arc_shape.draw_bezier(p0, p1, p2, p3)
        stroke_width = arc.get('stroke_width', 1)
        # Red color for all arcs
        arc_shape.finish(color=(1, 0, 0), width=stroke_width)
        arc_shape.commit()
        
        # Calculate arc center and draw coordinates next to it
        result = get_bezier_radius(control_points)
        if result:
            arc_radius, arc_center = result
            center_x = float(arc_center[0])
            center_y = float(arc_center[1])
            coord_text = f"({center_x:.0f}, {center_y:.0f})"
            # Position text offset from center
            text_x = center_x + 10
            text_y = center_y + 10
            # Draw coordinates
            new_page.insert_text(
                pymupdf.Point(text_x, text_y),
                coord_text,
                fontsize=8,
                color=(1, 0, 0)
            )

print(f"  Completed: {len(filtered_arcs)} arcs drawn in RED with coordinates ({len(original_arcs)} original, {len(reconstructed_arcs)} reconstructed)")

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_all_geometry.pdf"
new_doc.save(str(output_path))
new_doc.close()

print(f"\nOutput saved to: {output_path}")
