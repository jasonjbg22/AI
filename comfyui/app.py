# UPDATE:
# - Fixed Subfolder creation for 'heads' uploads
# - Improved Result Polling for Faceswap
# - Added robust error handling for uploads

import os
import json
import requests
import traceback
from datetime import datetime
from flask import Flask, request, jsonify

COMFY_IP = "100.89.240.18"
COMFY_PORT = "8188"
MAINT_PORT = "8189"

BASE_COMFY = f"http://{COMFY_IP}:{COMFY_PORT}"
BASE_MAINT = f"http://{COMFY_IP}:{MAINT_PORT}"

COMFY_UPLOAD = f"{BASE_COMFY}/upload/image"
COMFY_PROMPT = f"{BASE_COMFY}/prompt"
COMFY_HISTORY = f"{BASE_COMFY}/history"

MAINT_LIST = f"{BASE_MAINT}/list_outputs"
MAINT_FETCH = f"{BASE_MAINT}/fetch_image"
MAINT_DELETE = f"{BASE_MAINT}/delete_image"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
WORKFLOW_PATH = os.path.join(BASE_DIR, "workflows", "Edit Flux.json")
FACESWAP_PATH = os.path.join(BASE_DIR, "workflows", "Flux2 Faceswap.json")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

@app.route("/")
def index():
    return open(os.path.join(TEMPLATES_DIR, "index.html"), encoding="utf-8").read()

# ==========================================
#  GENERIC UPLOAD (To 'input' or subfolder)
# ==========================================
@app.route("/api/upload", methods=["POST"])
def upload_generic():
    try:
        file = request.files["image"]
        subfolder = request.form.get("subfolder", "")
        
        # Prepare the upload to ComfyUI
        files = {"image": (file.filename, file.stream, file.mimetype)}
        data = {"subfolder": subfolder} if subfolder else {}
        
        # ComfyUI will auto-create the subfolder in 'input/' if it doesn't exist
        r = requests.post(COMFY_UPLOAD, files=files, data=data, timeout=60)
        
        if r.status_code != 200:
            return jsonify({"status":"error","message":f"ComfyUI Error: {r.text}"}), 500
            
        # ComfyUI returns the filename it saved as (it might rename duplicates)
        saved_name = r.json().get("name", file.filename)
        
        return jsonify({
            "status": "success", 
            "filename": saved_name, 
            "subfolder": subfolder
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

# Legacy route
@app.route("/api/flux/upload", methods=["POST"])
def upload_legacy():
    return upload_generic()

# ==========================================
#  RUN FLUX (GENERATOR)
# ==========================================
@app.route("/api/flux/run", methods=["POST"])
def run_flux():
    try:
        data = request.json
        prompt = data.get("prompt","")
        image = data.get("image","")

        with open(WORKFLOW_PATH,"r",encoding="utf-8") as f:
            workflow=json.load(f)

        if "75:74" in workflow:
            workflow["75:74"]["inputs"]["text"]=prompt
        if "76" in workflow:
            workflow["76"]["inputs"]["image"]=image

        payload={"prompt":workflow}
        r=requests.post(COMFY_PROMPT,json=payload,timeout=120)

        if r.status_code!=200:
            return jsonify({"status":"error","message":r.text}),500

        return jsonify({"status":"queued","prompt_id":r.json().get("prompt_id")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

# ==========================================
#  RUN FACESWAP
# ==========================================
@app.route("/api/swap/run", methods=["POST"])
def run_swap():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        body_img = data.get("body_image", "")
        head_img = data.get("head_image", "")

        # Handle Library Heads:
        # If the head comes from the library, it's in the 'heads' subfolder.
        # We need to tell the LoadImage node that.
        # Standard LoadImage format for subfolders: "heads\filename.png" (Windows) or "heads/filename.png"
        
        if data.get("head_type") == "library":
             # Check if we need to prepend the folder
             if not head_img.startswith("heads") and "/" not in head_img and "\\" not in head_img:
                 head_img = f"heads\\{head_img}"

        with open(FACESWAP_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        if "151" in workflow:
            workflow["151"]["inputs"]["image"] = body_img

        if "121" in workflow:
            workflow["121"]["inputs"]["image"] = head_img

        if "107" in workflow:
            workflow["107"]["inputs"]["text"] = prompt

        payload = {"prompt": workflow}
        r = requests.post(COMFY_PROMPT, json=payload, timeout=120)

        if r.status_code != 200:
            return jsonify({"status": "error", "message": r.text}), 500

        return jsonify({"status": "queued", "prompt_id": r.json().get("prompt_id")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================
#  COMMON: POLL RESULT
# ==========================================
@app.route("/api/flux/result", methods=["GET"])
def result():
    try:
        pid = request.args.get("prompt_id")
        r = requests.get(COMFY_HISTORY, timeout=30)
        history = r.json()

        if pid not in history:
            return jsonify({"status":"pending"})

        # Get outputs for this prompt_id
        outputs = history[pid].get("outputs", {})
        
        # Scan all nodes for images
        for node_id, node_data in outputs.items():
            imgs = node_data.get("images")
            if imgs:
                # Use the first image found
                img = imgs[0]
                filename = img.get('filename')
                subfolder = img.get('subfolder', '')
                
                # Construct view URL
                url = f"{BASE_COMFY}/view?filename={filename}&subfolder={subfolder}&type=output"
                return jsonify({"status":"complete", "image_url":url})

        # If job is in history but has no images (cancelled or error), consider it done or error
        # For now, keep pending if we can't find image, or return error?
        # Let's return pending to be safe, or check status.
        return jsonify({"status":"pending"})
    except:
        return jsonify({"status":"error"})

# ==========================================
#  GALLERY & ASSETS
# ==========================================
@app.route("/api/flux/gallery", methods=["GET"])
def gallery():
    try:
        folder_type = request.args.get("type", "output")
        r = requests.get(f"{MAINT_LIST}?type={folder_type}", timeout=10)
        
        if r.status_code != 200:
            return jsonify({"status":"error"})
        
        files = r.json()
        return jsonify({"status":"success","images":[{"filename":f} for f in files]})
    except:
        return jsonify({"status":"error"})

@app.route("/api/flux/image", methods=["GET"])
def gallery_image():
    try:
        filename = request.args.get("filename")
        folder_type = request.args.get("type", "output")
        
        r = requests.get(f"{MAINT_FETCH}?filename={filename}&type={folder_type}", timeout=10)
        return jsonify(r.json())
    except:
        return jsonify({"status":"error"})

@app.route("/api/flux/delete", methods=["POST"])
def delete():
    try:
        filename = request.json.get("filename")
        folder_type = request.json.get("type", "output")
        
        r = requests.post(MAINT_DELETE, json={"filename": filename, "type": folder_type}, timeout=10)
        
        if r.status_code != 200:
            return jsonify({"status":"error"})
        return jsonify({"status":"success"})
    except:
        return jsonify({"status":"error"})

if __name__=="__main__":
    log("Flux Studio (Fixed Uploads + Progress) Starting...")
    app.run(host="0.0.0.0",port=5000,debug=True)
