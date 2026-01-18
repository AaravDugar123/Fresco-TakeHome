# Finds and annotates door on PDF

Door detection system for architectural floor plans using a vector-based geometric approach.

## How to run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `python server.py`
3. Open your browser: `http://localhost:5000`
4. Upload a PDF and click "Process PDF" to get an annotated PDF with door rectangles (red=swing, blue=double, green=bifold)


# Stats
Identifies three types of doors - Swing/Double/Bifold
80-90% accuracy for standard door types
Runs in any time from 10 seconds to 8 minutes depending on floor plans with most being under a minute
Minimizes False Positives as much as possible
Struggles with some edge cases & abstract floor plans