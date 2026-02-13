import os
import shutil
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# RAISE IMAGE LIMITS FOR 4K FLUX
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# CONFIRM THESE PATHS MATCH YOUR INSTALLATION
INPUT_DIR = r"C:\Comfyui\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\input"
OUTPUT_DIR = r"C:\Comfyui\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\output"
HEADS_DIR = os.path.join(INPUT_DIR, "heads")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HEADS_DIR, exist_ok=True)

@app.route('/list_outputs', methods=['GET'])
def list_out():
    try:
        folder_type = request.args.get("type", "output")
        target_dir = INPUT_DIR if folder_type == "input" else OUTPUT_DIR
        
        files = []
        if os.path.exists(target_dir):
            for root, _, filenames in os.walk(target_dir):
                for filename in filenames:
                    # Ignore hidden/system files
                    if filename.startswith('.'): continue
                    
                    # Only grab image files
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        continue

                    # Create a clean relative path (e.g., 'heads/myface.png')
                    rel_dir = os.path.relpath(root, target_dir)
                    if rel_dir == ".":
                        files.append(filename)
                    else:
                        # Force forward slashes so the frontend Javascript doesn't break
                        files.append(f"{rel_dir}/{filename}".replace("\\", "/"))
        
        # Sort by newest first
        files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, os.path.normpath(x))), reverse=True)
        return jsonify(files)
    except Exception as e:
        print(f"Error listing {folder_type}: {e}")
        return jsonify([])

@app.route('/fetch_image', methods=['GET'])
def fetch_img():
    filename = request.args.get('filename')
    folder_type = request.args.get('type', 'output')
    
    if not filename: return "Missing filename", 400
    
    target_dir = INPUT_DIR if folder_type == "input" else OUTPUT_DIR
    path = os.path.normpath(os.path.join(target_dir, filename))
    
    # Security check to prevent directory traversal
    if not path.startswith(os.path.abspath(target_dir)):
        return "Invalid path", 403

    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            
            ext = filename.split('.')[-1].lower()
            mime = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            return jsonify({"data": f"data:{mime};base64,{encoded}"})
        except:
            return "Read Error", 500
    
    return "File not found", 404

@app.route('/delete_image', methods=['POST', 'OPTIONS'])
def delete_img():
    if request.method == 'OPTIONS': return jsonify({"status": "ok"})
    
    data = request.json
    filename = data.get("filename")
    folder_type = data.get("type", "output")
    
    target_dir = INPUT_DIR if folder_type == "input" else OUTPUT_DIR
    
    # os.path.normpath ensures 'heads/image.png' resolves correctly on Windows
    file_path = os.path.normpath(os.path.join(target_dir, filename))
    
    # Security check to prevent directory traversal out of target_dir
    if not file_path.startswith(os.path.abspath(target_dir)):
        return jsonify({"status": "error", "message": "Invalid path"}), 403
        
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"status": "success", "message": "deleted"})
        
    return jsonify({"status": "error", "message": "File not found"}), 404

@app.route('/purge_all', methods=['POST', 'OPTIONS'])
def purge_all():
    if request.method == 'OPTIONS': return jsonify({"status": "ok"})
    
    deleted_count = 0
    
    # 1. Purge Outputs
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, f)
            try:
                if os.path.isfile(p):
                    os.unlink(p)
                    deleted_count += 1
            except: pass

    # 2. Purge Inputs (BUT SAVE HEADS)
    if os.path.exists(INPUT_DIR):
        for f in os.listdir(INPUT_DIR):
            # Skip the heads directory itself
            if f.lower() == 'heads': continue
            
            p = os.path.join(INPUT_DIR, f)
            try:
                if os.path.isfile(p):
                    os.unlink(p)
                    deleted_count += 1
            except: pass
            
    return jsonify({"status": "success", "deleted": deleted_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8189, debug=False)