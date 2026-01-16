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
    "AC_Convention_Center_ac_cc.pdf_-_Page_122.pdf"

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
door_result = classify_swing_doors(
    analysis['door_candidate_arcs'],  # Use pre-filtered arcs
    analysis['filtered_lines'],
    debug=True,  # Enable debug output
    page_width=result['page_width'],
    page_height=result['page_height']
)

swing_doors = door_result['swing_doors']
double_doors = door_result['double_doors']

print(f"\nFound {len(swing_doors)} swing doors, {len(double_doors)} double doors")

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

# Helper to get door bbox (use fast version from door_classifier)
from src.door_classifier import _get_door_bbox_fast as get_door_bbox

# Draw red rectangles around each swing door
for i, door in enumerate(swing_doors):
    min_x, min_y, max_x, max_y = get_door_bbox(door)
    padding = 10
    rect = pymupdf.Rect(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
    page.draw_rect(rect, color=(1, 0, 0), width=2)
    print(f"Swing door {i}: bbox=({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})")

# Draw blue rectangles around each double door
for i, double_door in enumerate(double_doors):
    min_x, min_y, max_x, max_y = double_door['bbox']
    
    # Add padding and draw
    padding = 10
    rect = pymupdf.Rect(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
    page.draw_rect(rect, color=(0, 0, 1), width=2)  # Blue color
    
    print(f"Double door {i}: bbox=({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})")

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
