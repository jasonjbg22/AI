@app.route("/api/flux/image", methods=["GET"])
def gallery_image():
    try:
        filename = request.args.get("filename")
        folder_type = request.args.get("type", "output")
        
        subfolder = ""
        comfy_type = folder_type
        
        if folder_type == "heads":
            comfy_type = "input"
            subfolder = "heads"
            
        url = f"{BASE_COMFY}/view?filename={filename}&type={comfy_type}&subfolder={subfolder}"
        r = requests.get(url, stream=True, timeout=10)
        
        if r.status_code == 200:
            return Response(r.raw.read(), mimetype=r.headers.get('content-type', 'image/png'))
        else:
            return jsonify({"status": "error"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route("/api/flux/image", methods=["GET"])
def gallery_image():
    try:
        filename = request.args.get("filename")
        folder_type = request.args.get("type", "output")
        
        url = f"{MAINT_FETCH}?filename={filename}&type={folder_type}"
        r = requests.get(url, stream=True, timeout=10)
        
        if r.status_code == 200:
            return Response(r.raw.read(), mimetype=r.headers.get('content-type', 'image/png'))
        else:
            return jsonify({"status": "error"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
