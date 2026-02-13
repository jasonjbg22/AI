# config.py
import os

# --- ⚙️ GLOBAL SETTINGS ---
FOLDER_MAP = {
    r"G:\AllPhotosDatabase\AI_ARt\em": "🎨 AI Art",
    r"G:\AllPhotosDatabase\2em":       "👩 Emily",
    r"G:\AllPhotosDatabase\Photos":    "👨‍👩‍👧‍👦 Family Photos",
}

# --- FILE SETTINGS (SSD) ---
DB_PATH = "./my_image_db"
FAV_PATH = "./favorites.json"
ALBUMS_PATH = "./smart_albums.json"
CAPTIONS_PATH = "./captions_db.json"
SCAN_MANIFEST = "./scan_manifest.json" 
THUMB_CACHE_DIR = "./neural_thumbs"

# --- AI SETTINGS ---
MODEL_NAME = "ViT-L/14"
DEVICE = "cuda"

# --- MOBILE UI SETTINGS ---
PRIMARY_COLOR = "#FF4B4B"
CARD_BG = "#1E1E1E"