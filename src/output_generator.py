"""Generate annotated floor plans with detected doors."""
import pymupdf
from typing import List, Dict


def draw_rejected_chains(pdf_path: str, output_path: str, rejected_chains: List[Dict], page_num: int = 0):
    """
    Draw boxes around rejected chains on a PDF and label them with coordinates.
    Creates a copy of the original PDF with red boxes and labels drawn on top.
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Path to save the annotated PDF
        rejected_chains: List of rejected chain dictionaries with 'bbox', 'center', 'chain_idx', 'reason'
        page_num: Page number to annotate (default 0)
    """
    # Open the original PDF
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    
    # Draw boxes around each rejected chain
    for chain_info in rejected_chains:
        bbox = chain_info['bbox']  # (min_x, min_y, max_x, max_y)
        center = chain_info['center']  # (x, y)
        chain_idx = chain_info['chain_idx']
        reason = chain_info.get('reason', 'unknown')
        
        # Create a shape for the box
        shape = page.new_shape()
        
        # Draw rectangle (bbox format: x0, y0, x1, y1)
        # PyMuPDF uses bottom-left origin, so coordinates are correct as-is
        rect = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        shape.draw_rect(rect)
        
        # Draw in red with 2pt width
        shape.finish(color=(1, 0, 0), width=2.0)
        shape.commit()
        
        # Add text label with coordinates and chain index
        # Position label above the box (or below if near top of page)
        page_height = page.rect.height
        if bbox[3] + 30 < page_height:
            # Label above the box
            label_y = bbox[3] + 10
        else:
            # Label below the box
            label_y = bbox[1] - 5
        
        label_x = bbox[0]  # Left edge of box
        
        # Create text annotation with chain index and coordinates
        coord_text = f"#{chain_idx}: ({center[0]:.1f}, {center[1]:.1f})"
        
        # Insert text
        point = pymupdf.Point(label_x, label_y)
        page.insert_text(
            point,
            coord_text,
            fontsize=8,
            color=(1, 0, 0),  # Red text
            render_mode=0  # Fill text
        )
        
        # Add a shorter reason on the next line if it fits
        reason_short = reason.split('(')[0] if '(' in reason else reason[:25]
        if len(reason_short) > 0:
            reason_y = label_y + 10 if bbox[3] + 30 < page_height else label_y - 10
            page.insert_text(
                pymupdf.Point(label_x, reason_y),
                reason_short[:25],
                fontsize=6,
                color=(0.8, 0, 0),  # Darker red
                render_mode=0
            )
    
    # Save the annotated PDF
    doc.save(output_path)
    doc.close()
    
    print(f"Drew {len(rejected_chains)} rejected chain boxes on PDF")
    print(f"Output saved to: {output_path}")
