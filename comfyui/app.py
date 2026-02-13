import os
import json
import requests
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, Response

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
MAINT_COPY = f"{BASE_MAINT}/copy_to_input"  # <--- NEW ROUTE ADDED HERE

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
        
        files = {"image": (file.filename, file.stream, file.mimetype)}
        data = {"subfolder": subfolder} if subfolder else {}
        
        r = requests.post(COMFY_UPLOAD, files=files, data=data, timeout=60)
        
        if r.status_code != 200:
            return jsonify({"status":"error","message":f"ComfyUI Error: {r.text}"}), 500
            
        saved_name = r.json().get("name", file.filename)
        
        return jsonify({
            "status": "success", 
            "filename": saved_name, 
            "subfolder": subfolder
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

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

        if data.get("head_type") == "library":
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

        outputs = history[pid].get("outputs", {})
        
        for node_id, node_data in outputs.items():
            imgs = node_data.get("images")
            if imgs:
                img = imgs[0]
                filename = img.get('filename')
                subfolder = img.get('subfolder', '')
                
                url = f"{BASE_COMFY}/view?filename={filename}&subfolder={subfolder}&type=output"
                return jsonify({"status":"complete", "image_url":url})

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
        
        if folder_type == "heads":
            r = requests.get(f"{MAINT_LIST}?type=input", timeout=10)
            if r.status_code != 200: return jsonify({"status": "error"})
            files = [f for f in r.json() if f.startswith("heads/") or f.startswith("heads\\")]
            clean_files = [f.replace("heads/", "").replace("heads\\", "") for f in files]
            return jsonify({"status":"success","images":[{"filename":f} for f in clean_files]})
            
        elif folder_type == "input":
            r = requests.get(f"{MAINT_LIST}?type=input", timeout=10)
            if r.status_code != 200: return jsonify({"status": "error"})
            files = [f for f in r.json() if not f.startswith("heads/") and not f.startswith("heads\\")]
            return jsonify({"status":"success","images":[{"filename":f} for f in files]})
            
        else:
            r = requests.get(f"{MAINT_LIST}?type=output", timeout=10)
            if r.status_code != 200: return jsonify({"status":"error"})
            return jsonify({"status":"success","images":[{"filename":f} for f in r.json()]})
            
    except:
        return jsonify({"status":"error"})

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

# ==========================================
#  COPY & DELETE
# ==========================================
@app.route("/api/flux/copy", methods=["POST"])
def copy_file_to_input():
    try:
        filename = request.json.get("filename")
        # Tell the sidecar maint.py to execute the double-copy logic
        r = requests.post(MAINT_COPY, json={"filename": filename}, timeout=10)
        
        if r.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to copy on maint server"}), 500
        return jsonify({"status": "success"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/flux/delete", methods=["POST"])
def delete():
    try:
        filename = request.json.get("filename")
        folder_type = request.json.get("type", "output")
        
        if folder_type == "heads":
            filename = f"heads/{filename}"
            folder_type = "input"
            
        r = requests.post(MAINT_DELETE, json={"filename": filename, "type": folder_type}, timeout=10)
        
        if r.status_code != 200:
            return jsonify({"status":"error"})
        return jsonify({"status":"success"})
    except:
        return jsonify({"status":"error"})

@app.route("/api/flux/purge", methods=["POST"])
def purge_all():
    try:
        deleted = 0
        
        # Purge outputs
        r_out = requests.get(f"{MAINT_LIST}?type=output", timeout=10)
        if r_out.status_code == 200:
            for f in r_out.json():
                requests.post(MAINT_DELETE, json={"filename": f, "type": "output"}, timeout=5)
                deleted += 1
                
        # Purge inputs (excluding heads)
        r_in = requests.get(f"{MAINT_LIST}?type=input", timeout=10)
        if r_in.status_code == 200:
            for f in r_in.json():
                if not f.startswith("heads/") and not f.startswith("heads\\"):
                    requests.post(MAINT_DELETE, json={"filename": f, "type": "input"}, timeout=5)
                    deleted += 1
                    
        return jsonify({"status": "success", "data": {"deleted": deleted}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__=="__main__":
    log("Flux Studio Starting...")
    app.run(host="0.0.0.0",port=5000,debug=True)