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
@app.route('/fetch_image', methods=['GET'])
def fetch_img():
    filename = request.args.get('filename')
    folder_type = request.args.get('type', 'output')
    
    if not filename: return "Missing filename", 400
    
    target_dir = INPUT_DIR if folder_type in ["input", "heads"] else OUTPUT_DIR
    
    if folder_type == "heads" and not filename.startswith("heads"):
        filename = os.path.join("heads", filename)
        
    path = os.path.normpath(os.path.join(target_dir, filename))
    
    # Security check to prevent directory traversal
    if not path.startswith(os.path.abspath(target_dir)):
        return "Invalid path", 403

    if os.path.exists(path):
        return send_file(path)
    
    return "File not found", 404

@app.route('/copy_to_input', methods=['POST', 'OPTIONS'])
def copy_to_input():
    if request.method == 'OPTIONS': return jsonify({"status": "ok"})
    
    filename = request.json.get("filename")
    if not filename: return jsonify({"status": "error", "message": "No filename"}), 400
    
    src = os.path.normpath(os.path.join(OUTPUT_DIR, filename))
    dst = os.path.normpath(os.path.join(INPUT_DIR, filename))
    
    if not src.startswith(os.path.abspath(OUTPUT_DIR)):
        return jsonify({"status": "error", "message": "Invalid path"}), 403
        
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    return jsonify({"status": "ignored"})
