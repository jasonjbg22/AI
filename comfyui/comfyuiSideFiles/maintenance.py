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

def get_target_dir(folder_type):
    if folder_type == 'heads': return HEADS_DIR
    if folder_type == 'input': return INPUT_DIR
    return OUTPUT_DIR

@app.route('/list_outputs', methods=['GET'])
def list_out():
    try:
        folder_type = request.args.get('type', 'output')
        target_dir = get_target_dir(folder_type)
        
        if not os.path.exists(target_dir):
            return jsonify([])

        # Gather files
        files = []
        for f in os.listdir(target_dir):
            # Filter out the 'heads' folder if listing generic inputs
            if folder_type == 'input' and f == 'heads':
                continue
                
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                files.append(f)

        # Sort by newest first
        files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
        return jsonify(files)
    except Exception as e:
        print(f"Error listing {folder_type}: {e}")
        return jsonify([]) # Return empty list on error instead of 500

@app.route('/fetch_image', methods=['GET'])
def fetch_img():
    filename = request.args.get('filename')
    folder_type = request.args.get('type', 'output')
    
    if not filename: return "Missing filename", 400
    
    target_dir = get_target_dir(folder_type)
    path = os.path.join(target_dir, filename)
    
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
    filename = data.get('filename')
    folder_type = data.get('type', 'output')
    
    target_dir = get_target_dir(folder_type)
    path = os.path.join(target_dir, filename)
    
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404

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
            if f == 'heads': continue # SKIP HEADS
            
            p = os.path.join(INPUT_DIR, f)
            try:
                if os.path.isfile(p):
                    os.unlink(p)
                    deleted_count += 1
            except: pass
            
    return jsonify({"status": "success", "deleted": deleted_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8189, debug=False)
