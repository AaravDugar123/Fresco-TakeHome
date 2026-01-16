"""Visualize rejected chains with boxes and detailed information."""
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

# Analyze geometry to get rejected chains
print("\nAnalyzing geometry...")
analysis = analyze_geometry(
    result['lines'],
    result['arcs'],
    result['dashed_lines'],
    result['page_width'],
    result['page_height']
)

rejected_chains = analysis.get('rejected_chains', [])

if not rejected_chains:
    print("\nNo rejected chains to visualize.")
    sys.exit(0)

print(f"\nFound {len(rejected_chains)} rejected chains")

# Open the original PDF
doc = pymupdf.open(pdf_path)
page = doc[0]
page_height = page.rect.height
page_width = result['page_width']

# Add coordinate labels on bottom and left edges (no grid lines)
print("\nAdding coordinate labels...")
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

# Draw boxes around each rejected chain with coordinates only
print("\nDrawing rejected chain boxes with coordinates...")
for chain_info in rejected_chains:
    bbox = chain_info['bbox']  # (min_x, min_y, max_x, max_y)
    center = chain_info['center']  # (x, y)

    # Create a shape for the box
    shape = page.new_shape()

    # Draw rectangle (bbox format: x0, y0, x1, y1)
    rect = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    shape.draw_rect(rect)

    # Draw in red with 2pt width
    shape.finish(color=(1, 0, 0), width=2.0)
    shape.commit()

    # Determine label position (above or below box)
    if bbox[3] + 30 < page_height:
        # Label above the box
        label_y = bbox[3] + 10
    else:
        # Label below the box
        label_y = bbox[1] - 5

    label_x = bbox[0]  # Left edge of box

    # Draw only coordinates
    coord_text = f"({center[0]:.1f}, {center[1]:.1f})"
    page.insert_text(
        pymupdf.Point(label_x, label_y),
        coord_text,
        fontsize=9,
        color=(1, 0, 0),  # Red text
        render_mode=0
    )

# Save output
output_dir = Path(__file__).parent.parent / "Data" / "Output_drawings"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}_rejected_chains.pdf"
doc.save(str(output_path))
doc.close()

print(f"\nRejected chains visualization saved to: {output_path}")
print(f"Total rejected chains: {len(rejected_chains)}")

# Print summary
print("\n=== REJECTION SUMMARY ===")
rejection_reasons = {}
for chain_info in rejected_chains:
    reason_type = chain_info['reason'].split(
        '(')[0] if '(' in chain_info['reason'] else chain_info['reason']
    rejection_reasons[reason_type] = rejection_reasons.get(reason_type, 0) + 1

for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"  {reason}: {count} chains")
