"""Test radius and sweep angle calculations on known curves."""
import sys
from pathlib import Path
import pymupdf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry_analyzer import get_bezier_radius
from src.door_classifier import calculate_arc_sweep_angle

# Create a test PDF with known curves
doc = pymupdf.open()  # Create new PDF
page = doc.new_page(width=800, height=600)

# Test 1: Draw a quarter circle (90 degree arc) with known radius
# We'll draw it as a cubic Bezier curve approximating a quarter circle
radius = 50.0
center_x, center_y = 100, 100

# For a quarter circle from (0, radius) to (radius, 0) centered at origin:
# Control points for cubic Bezier approximating quarter circle
# Using the standard approximation: control point distance = radius * 4/3 * tan(angle/4)
# For 90 degrees: tan(22.5°) ≈ 0.414, so control distance ≈ radius * 0.552

p0 = (center_x, center_y - radius)  # Start: top
p1 = (center_x + radius * 0.552, center_y - radius)  # Control 1
p2 = (center_x + radius, center_y - radius * 0.552)  # Control 2
p3 = (center_x + radius, center_y)  # End: right

# Draw the curve
page.draw_bezier(p0, p1, p2, p3, color=(0, 0, 0), width=1)

# Test 2: Draw a semicircle (180 degree arc)
radius2 = 40.0
center_x2, center_y2 = 300, 100

# Semicircle from left to right
p0_2 = (center_x2 - radius2, center_y2)
p1_2 = (center_x2 - radius2, center_y2 - radius2 * 0.552)
p2_2 = (center_x2 + radius2, center_y2 - radius2 * 0.552)
p3_2 = (center_x2 + radius2, center_y2)

page.draw_bezier(p0_2, p1_2, p2_2, p3_2, color=(0, 0, 0), width=1)

# Test 3: Draw a small door-like arc (typical door swing ~90 degrees, radius ~30-40)
radius3 = 35.0
center_x3, center_y3 = 500, 100

p0_3 = (center_x3, center_y3 - radius3)
p1_3 = (center_x3 + radius3 * 0.552, center_y3 - radius3)
p2_3 = (center_x3 + radius3, center_y3 - radius3 * 0.552)
p3_3 = (center_x3 + radius3, center_y3)

page.draw_bezier(p0_3, p1_3, p2_3, p3_3, color=(0, 0, 0), width=1)

# Save test PDF
test_pdf_path = Path(__file__).parent.parent / "Data" / "test_curves.pdf"
test_pdf_path.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(test_pdf_path))
doc.close()

print(f"Created test PDF with curves at: {test_pdf_path}\n")

# Now extract and analyze the curves
doc = pymupdf.open(str(test_pdf_path))
page = doc[0]

# Extract curves using get_cdrawings
paths = page.get_cdrawings()
arcs = []

for path in paths:
    path_type = path.get("type", "")
    if path_type == "f":  # fill-only, skip
        continue
    
    for item in path["items"]:
        if item[0] == "c":  # cubic Bezier curve
            p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
            arc_data = {
                "type": "cubic_bezier",
                "control_points": [p0, p1, p2, p3],
                "stroke_width": path.get("width", 1)
            }
            arcs.append(arc_data)

print(f"Found {len(arcs)} curves in test PDF\n")

# Test radius and sweep angle calculations
for i, arc in enumerate(arcs):
    print(f"--- Curve {i+1} ---")
    print(f"Control points:")
    for j, pt in enumerate(arc['control_points']):
        print(f"  P{j}: ({pt[0]:.2f}, {pt[1]:.2f})")
    
    # Calculate radius
    result = get_bezier_radius(arc['control_points'])
    if result is None:
        print("  Radius calculation: FAILED (collinear points)")
    else:
        calculated_radius, midpoint = result
        print(f"  Calculated radius: {calculated_radius:.2f}")
        print(f"  Midpoint: ({midpoint[0]:.2f}, {midpoint[1]:.2f})")
        
        # Expected radius (based on which test curve)
        if i == 0:
            expected_radius = 50.0
        elif i == 1:
            expected_radius = 40.0
        else:
            expected_radius = 35.0
        
        error = abs(calculated_radius - expected_radius) / expected_radius * 100
        print(f"  Expected radius: {expected_radius:.2f}")
        print(f"  Error: {error:.1f}%")
    
    # Calculate sweep angle
    sweep_angle = calculate_arc_sweep_angle(arc)
    if sweep_angle is None:
        print("  Sweep angle: FAILED")
    else:
        print(f"  Calculated sweep angle: {sweep_angle:.1f} degrees")
        
        # Expected sweep angle (all are ~90 degrees)
        expected_angle = 90.0
        angle_error = abs(sweep_angle - expected_angle)
        print(f"  Expected sweep angle: {expected_angle:.1f} degrees")
        print(f"  Error: {angle_error:.1f} degrees")
    
    print()

doc.close()
