from flask import Flask, request, render_template, send_from_directory, jsonify
import cv2
import numpy as np
import os
import uuid
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, output_path, edge_threshold1=50, edge_threshold2=150, 
                  hough_threshold=100, min_line_length=100, max_line_gap=5,
                  dilation_iterations=2, inpaint_radius=3):
    image = cv2.imread(image_path)
    if image is None:
        return False, "Could not read image file"
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      
        edges = cv2.Canny(gray, edge_threshold1, edge_threshold2, apertureSize=3)

        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold, 
                               minLineLength=min_line_length, maxLineGap=max_line_gap)

        
        line_mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
        
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        lines_viz = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        kernel = np.ones((5, 5), np.uint8)
        line_mask_dilated = cv2.dilate(line_mask, kernel, iterations=dilation_iterations)

        mask_colored = cv2.cvtColor(line_mask_dilated, cv2.COLOR_GRAY2BGR)
        
        image_no_pipes = cv2.inpaint(image, line_mask_dilated, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        
        cv2.imwrite(output_path, image_no_pipes)
        
        edges_path = output_path.replace('processed_', 'edges_')
        lines_path = output_path.replace('processed_', 'lines_')
        mask_path = output_path.replace('processed_', 'mask_')
        
        cv2.imwrite(edges_path, edges_colored)
        cv2.imwrite(lines_path, lines_viz)
        cv2.imwrite(mask_path, mask_colored)
        
        return True, {
            "processed": os.path.basename(output_path),
            "edges": os.path.basename(edges_path),
            "lines": os.path.basename(lines_path),
            "mask": os.path.basename(mask_path)
        }
    except Exception as e:
        return False, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/before-after')
def before_after():
    return render_template('after&before.html')

@app.route('/api/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename_parts = os.path.splitext(filename)
        unique_filename = f"{filename_parts[0]}_{uuid.uuid4().hex[:8]}{filename_parts[1]}"
        
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        output_path = os.path.join(RESULT_FOLDER, 'processed_' + unique_filename)
        file.save(input_path)
        
        edge_threshold1 = int(request.form.get('edgeThreshold1', 50))
        edge_threshold2 = int(request.form.get('edgeThreshold2', 150))
        hough_threshold = int(request.form.get('houghThreshold', 100))
        min_line_length = int(request.form.get('minLineLength', 100))
        max_line_gap = int(request.form.get('maxLineGap', 5))
        dilation_iterations = int(request.form.get('dilationIterations', 2))
        inpaint_radius = int(request.form.get('inpaintRadius', 3))
        
        time.sleep(1)
        
        success, result = process_image(
            input_path, output_path,
            edge_threshold1, edge_threshold2,
            hough_threshold, min_line_length, max_line_gap,
            dilation_iterations, inpaint_radius
        )
        
        if success:
            return jsonify({
                "success": True,
                "inputImage": unique_filename,
                "results": result
            })
        else:
            return jsonify({"success": False, "error": result})
    
    return jsonify({"success": False, "error": "Invalid file format"})

@app.route('/api/parameters', methods=['GET'])
def get_default_parameters():
    return jsonify({
        "edgeThreshold1": 50,
        "edgeThreshold2": 150,
        "houghThreshold": 100,
        "minLineLength": 100,
        "maxLineGap": 5,
        "dilationIterations": 2,
        "inpaintRadius": 3
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def processed_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)