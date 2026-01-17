"""Extract geometric primitives/shapes from PDF using PyMuPDF."""
import pymupdf
from typing import List, Dict, Tuple


def extract_vectors(pdf_path: str, page_num: int = 0) -> Dict:
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]

    # Get page dimensions
    page_width = page.rect.width
    page_height = page.rect.height

    # Extract primitives
    lines, arcs, dashed_lines = extract_primitives(page)

    doc.close()

    return {
        "lines": lines,
        "arcs": arcs,
        "dashed_lines": dashed_lines,
        "page_width": page_width,
        "page_height": page_height
    }


def extract_primitives(page: pymupdf.Page) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Extract all geometric primitives from a PDF page.
    Returns
        Tuple of (lines, arcs, dashed_lines)
    """
    paths = page.get_cdrawings()  # signficantly faster than just get_drawing

    lines = []
    arcs = []
    dashed_lines = []

    for path in paths:
        # only process stroke paths
        path_type = path.get("type", "")
        if path_type == "f":  # skip filled in shapes
            continue

        stroke_width = path.get("width", 1)
        is_dashed = path.get("dashes") is not None
        path_rect = path.get("rect")  # Bounding box of entire path
        close_path = path.get("closePath", False)  # Check if path is closed (circle)

        for item in path["items"]:
            if item[0] == "l":  # line
                line_data = {
                    "start": item[1],
                    "end": item[2],
                    "stroke_width": stroke_width,
                    "is_dashed": is_dashed,
                    "path_rect": path_rect  #
                }

                if is_dashed:
                    dashed_lines.append(line_data)
                else:
                    lines.append(line_data)

            elif item[0] == "c":  # cubic Bezier curve (4 points)
                p0, p1, p2, p3 = item[1], item[2], item[3], item[4]

                arc_data = {
                    "type": "cubic_bezier",
                    "control_points": [p0, p1, p2, p3],  
                    "stroke_width": stroke_width,
                    "path_rect": path_rect,
                    "close_path": close_path  # Store if path is closed (circle) PyMuPDf doesnt process this right
                }
                arcs.append(arc_data)

    return lines, arcs, dashed_lines
