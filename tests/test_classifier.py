"""Test to find swing doors and draw red rectangles around them."""
from src.door_classifier import classify_swing_doors
from src.geometry_analyzer import analyze_geometry
from src.vector_extractor import extract_vectors
import sys
import time
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).parent.parent))


# Path relative to project root
pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "FirstSource_R25-01360-A-V03.pdf_-_Page_2.pdf"

# Start timing
start_time = time.time()

# Extract vectors
print("Extracting vectors...")
result = extract_vectors(str(pdf_path))

# Analyze geometry
print("Analyzing geometry...")
analysis = analyze_geometry(
    result['lines'],
    result['arcs'],
    result['dashed_lines'],
    result['page_width'],
    result['page_height']
)

# Classify swing doors (use pre-filtered candidates to avoid redundant checks)
print("Classifying swing doors...")
swing_doors = classify_swing_doors(
    analysis['door_candidate_arcs'],  # Use pre-filtered arcs
    analysis['filtered_lines'],
    debug=True  # Enable debug output
)

print(f"\nFound {len(swing_doors)} swing doors")

# Open PDF and draw rectangles
doc = pymupdf.open(str(pdf_path))
page = doc[0]

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
        page.insert_text(
            pymupdf.Point(x, label_offset),
            str(x),
            fontsize=font_size,
            color=(0.5, 0.5, 0.5)
        )

# Y-axis labels (left edge) - PDF origin is bottom-left
for y in range(0, int(page_height) + label_spacing, label_spacing):
    if y <= page_height - 10:  # Ensure label fits within page
        page.insert_text(
            pymupdf.Point(label_offset, y),
            str(y),
            fontsize=font_size,
            color=(0.5, 0.5, 0.5)
        )

print(f"  Coordinate labels added: {label_spacing} unit spacing")

# Draw red rectangles around each swing door
for i, door in enumerate(swing_doors):
    arc = door['arc']
    line = door['line']

    # Get bounding boxes from path_rect or calculate from points
    arc_rect = arc.get('path_rect')
    line_rect = line.get('path_rect')

    # Extract bbox from arc
    if arc_rect:
        if isinstance(arc_rect, pymupdf.Rect):
            arc_bbox = (arc_rect.x0, arc_rect.y0, arc_rect.x1, arc_rect.y1)
        else:
            arc_bbox = arc_rect
    else:
        # Fallback: calculate from control points
        control_points = arc['control_points']
        all_x = [p[0] for p in control_points]
        all_y = [p[1] for p in control_points]
        arc_bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    # Extract bbox from line
    if line_rect:
        if isinstance(line_rect, pymupdf.Rect):
            line_bbox = (line_rect.x0, line_rect.y0,
                         line_rect.x1, line_rect.y1)
        else:
            line_bbox = line_rect
    else:
        # Fallback: calculate from start/end points
        start = line['start']
        end = line['end']
        line_bbox = (min(start[0], end[0]), min(start[1], end[1]),
                     max(start[0], end[0]), max(start[1], end[1]))

    # Find combined bounding box
    min_x = min(arc_bbox[0], line_bbox[0])
    min_y = min(arc_bbox[1], line_bbox[1])
    max_x = max(arc_bbox[2], line_bbox[2])
    max_y = max(arc_bbox[3], line_bbox[3])

    # Add padding
    padding = 10
    rect = pymupdf.Rect(
        min_x - padding,
        min_y - padding,
        max_x + padding,
        max_y + padding
    )

    # Draw red rectangle
    page.draw_rect(rect, color=(1, 0, 0), width=2)  # Red color

    print(
        f"Swing door {i}: bbox=({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})")

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_swing_doors.pdf"
doc.save(str(output_path))
doc.close()

print(f"\nOutput saved to: {output_path}")

# Calculate and print time as the last thing
total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f} seconds")
