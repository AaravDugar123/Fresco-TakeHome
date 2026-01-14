"""Simple test to print extracted vectors from a PDF."""
from src.vector_extractor import extract_vectors
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))


# Path relative to project root (go up from tests/ to project root)
pdf_path = Path(__file__).parent.parent / "Data" / "door_drawings" / \
    "FirstSource_R25-01360-A-V03.pdf_-_Page_2.pdf"

result = extract_vectors(str(pdf_path))  # Convert back to string for PyMuPDF

print("=== LINES ===")
for i, line in enumerate(result['lines']):
    print(
        f"Line {i}: start={line['start']}, end={line['end']}, stroke_width={line['stroke_width']}")

print("\n=== DASHED LINES ===")
for i, line in enumerate(result['dashed_lines']):
    print(
        f"Dashed Line {i}: start={line['start']}, end={line['end']}, stroke_width={line['stroke_width']}")

print("\n=== ARCS/CURVES ===")
for i, arc in enumerate(result['arcs']):
    print(
        f"Arc {i}: control_points={arc['control_points']}, stroke_width={arc['stroke_width']}")

print(f"\nPage size: {result['page_width']} x {result['page_height']}")
