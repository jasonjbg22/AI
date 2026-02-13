import os
from flask import Flask, render_template, send_from_directory, jsonify, request

app = Flask(__name__)

# External directory containing the 100k+ assets
THUMB_DIR = r"C:\Users\Jason\Documents\trae_projects\Gallery\neural_thumbs"
PER_PAGE = 50

# In-memory cache to prevent slow os.listdir calls on every page load
FILE_CACHE = []
CACHE_LOADED = False

def get_all_files():
    global FILE_CACHE, CACHE_LOADED
    if not CACHE_LOADED:
        print("Loading file list into memory. This might take a few seconds for 100k+ files...")
        try:
            if not os.path.exists(THUMB_DIR):
                print(f"ERROR: Directory not found: {THUMB_DIR}")
                return []
                
            files = [f for f in os.listdir(THUMB_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.webm', '.gif'))]
            
            # Sort by modification time (newest first)
            print("Sorting files by date...")
            files.sort(key=lambda x: os.path.getmtime(os.path.join(THUMB_DIR, x)), reverse=True)
            
            FILE_CACHE = files
            CACHE_LOADED = True
            print(f"Successfully loaded {len(FILE_CACHE)} files into cache.")
        except Exception as e:
            print(f"Error reading directory: {e}")
            return []
    return FILE_CACHE

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/thumbs')
def get_thumbs():
    page = int(request.args.get('page', 0))
    all_files = get_all_files()
    
    start = page * PER_PAGE
    end = start + PER_PAGE
    paged_files = all_files[start:end]
    
    return jsonify({
        "files": paged_files,
        "has_more": end < len(all_files),
        "total": len(all_files)
    })

@app.route('/file/<path:filename>')
def serve_file(filename):
    return send_from_directory(THUMB_DIR, filename)

@app.route('/api/refresh')
def refresh_cache():
    global CACHE_LOADED
    CACHE_LOADED = False
    return jsonify({"status": "Cache cleared, will reload on next request."})

if __name__ == "__main__":
    print(f"Starting Neural Thumbs Gallery Server at http://localhost:5001")
    app.run(port=5001, debug=True)
