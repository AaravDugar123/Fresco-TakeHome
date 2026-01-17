"""Simple Flask web app for PDF door classification."""
from flask import Flask, request, send_file, render_template_string, jsonify
from flask_cors import CORS
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.door_classifier import classify_swing_doors, _get_door_bbox_fast
from src.geometry_analyzer import analyze_geometry
from src.vector_extractor import extract_vectors
import pymupdf

app = Flask(__name__)
CORS(app)

HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PDF Door Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
    <div class="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
        <h1 class="text-2xl font-bold mb-6">PDF Door Classifier</h1>
        <input type="file" accept=".pdf" id="fileInput" 
               class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 mb-4">
        <p id="fileName" class="text-sm text-gray-600 mb-4"></p>
        <button onclick="processPDF()" id="btn" 
                class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded">
            Process PDF
        </button>
        <div id="status" class="mt-4"></div>
    </div>
    <script>
        let file = null;
        document.getElementById('fileInput').addEventListener('change', (e) => {
            file = e.target.files[0];
            document.getElementById('fileName').textContent = file ? `Selected: ${file.name}` : '';
            document.getElementById('status').innerHTML = '';
        });
        async function processPDF() {
            if (!file) { document.getElementById('status').innerHTML = '<p class="text-red-600">Please select a file</p>'; return; }
            const btn = document.getElementById('btn');
            btn.disabled = true;
            btn.textContent = 'Processing...';
            document.getElementById('status').innerHTML = '<p class="text-blue-600">Processing PDF...</p>';
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await fetch('/process-pdf', { method: 'POST', body: formData });
                if (!res.ok) {
                    let errorMsg = 'Failed to process PDF';
                    try {
                        const contentType = res.headers.get('content-type');
                        if (contentType && contentType.includes('application/json')) {
                            const err = await res.json();
                            errorMsg = err.error || errorMsg;
                        } else {
                            errorMsg = `Server error: ${res.status} ${res.statusText}`;
                        }
                    } catch (e) {
                        errorMsg = `Server error: ${res.status} ${res.statusText}`;
                    }
                    throw new Error(errorMsg);
                }
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'annotated.pdf';
                a.click();
                window.URL.revokeObjectURL(url);
                document.getElementById('status').innerHTML = '<p class="text-green-600">Success! Download started.</p>';
            } catch (e) {
                document.getElementById('status').innerHTML = `<p class="text-red-600">${e.message}</p>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Process PDF';
            }
        }
    </script>
</body>
</html>"""


@app.route('/')
def index():
    return HTML


@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_input:
        file.save(tmp_input.name)
        input_path = tmp_input.name
    
    output_path = None
    try:
        # Extract vectors (same as test_classifier.py)
        result = extract_vectors(input_path)
        
        # Analyze geometry (same as test_classifier.py)
        analysis = analyze_geometry(
            result['lines'],
            result['arcs'],
            result['dashed_lines'],
            result['page_width'],
            result['page_height']
        )
        
        # Classify doors (same as test_classifier.py)
        door_result = classify_swing_doors(
            analysis['door_candidate_arcs'],
            analysis['filtered_lines'],
            debug=False,
            page_width=result['page_width'],
            page_height=result['page_height'],
            double_door_candidates=analysis.get('double_door_candidates', [])
        )
        
        swing_doors = door_result['swing_doors']
        double_doors = door_result['double_doors']
        bifold_doors = door_result.get('bifold_doors', [])
        
        # Open PDF and draw rectangles (same as test_classifier.py)
        doc = pymupdf.open(input_path)
        page = doc[0]
        
        # Draw red rectangles for swing doors
        for door in swing_doors:
            min_x, min_y, max_x, max_y = _get_door_bbox_fast(door)
            padding = 10
            rect = pymupdf.Rect(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
            page.draw_rect(rect, color=(1, 0, 0), width=2)
        
        # Draw blue rectangles for double doors
        for double_door in double_doors:
            min_x, min_y, max_x, max_y = double_door['bbox']
            padding = 10
            rect = pymupdf.Rect(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
            page.draw_rect(rect, color=(0, 0, 1), width=2)
        
        # Draw green rectangles for bifold doors
        for bifold_door in bifold_doors:
            min_x, min_y, max_x, max_y = bifold_door['bbox']
            padding = 10
            rect = pymupdf.Rect(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
            page.draw_rect(rect, color=(0, 1, 0), width=2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
            output_path = tmp_output.name
            doc.save(output_path)
            doc.close()
        
        return send_file(output_path, as_attachment=True, download_name='annotated.pdf', mimetype='application/pdf')
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        if app.debug:
            error_msg += '\n' + traceback.format_exc()
        return jsonify({'error': error_msg}), 500
    
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.unlink(input_path)


if __name__ == '__main__':
    print("=" * 50)
    print("Starting PDF Door Classifier Server")
    print("Open your browser: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
