import streamlit as st
import chromadb
import clip
import torch
import os
import time
import json
import random
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ExifTags
from transformers import BlipProcessor, BlipForConditionalGeneration

# Try-Import for Advanced Math
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Neural Gallery Ultimate", page_icon="🧠")

# Try Clipboard
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

# Paths
FOLDER_MAP = {
    r"G:\AllPhotosDatabase\AI_ARt\em": "🎨 AI Art",
    r"G:\AllPhotosDatabase\2em":       "👩 Emily Only",
    r"G:\AllPhotosDatabase\Photos":    "👨‍👩‍👧‍👦 Family Photos",
}
SEARCH_FOLDERS = list(FOLDER_MAP.keys())
DB_PATH = "./my_image_db"
FAV_PATH = "./favorites.json"
ALBUMS_PATH = "./smart_albums.json"
CAPTIONS_PATH = "./captions_db.json"
MODEL_NAME = "ViT-L/14"

# --- 2. CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stButton"] button { border-radius: 8px; border: 1px solid #444; text-align: left; transition: 0.2s; }
    div[data-testid="stButton"] button:hover { border-color: #666; background-color: #333; }
    .tag-badge { background-color: #333; color: #eee; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 5px; border: 1px solid #555; }
    .ai-bubble { background-color: #2b313e; padding: 10px; border-radius: 10px; border-left: 3px solid #007bff; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. RESOURCE LOADING ---
@st.cache_resource
def load_search_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    return model, preprocess, device

@st.cache_resource
def load_gen_ai():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    return processor, model, device

@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="my_images")
    return collection

try:
    clip_model, clip_preprocess, device = load_search_engine()
    collection = load_database()
    blip_processor, blip_model, _ = load_gen_ai()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- 4. STATE ---
def init_state():
    defaults = {
        'display_items': [], 'viewing_item': None, 'last_text_query': "", 
        'sort_order': "AI Similarity", 'on_this_day_mode': False,
        'history': [], 'favorites': set(), 'smart_albums': {}, 'captions': {},
        'duplicate_candidates': []
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    
    if not st.session_state.favorites and os.path.exists(FAV_PATH):
        with open(FAV_PATH, 'r') as f: st.session_state.favorites = set(json.load(f))
    if not st.session_state.smart_albums and os.path.exists(ALBUMS_PATH):
        with open(ALBUMS_PATH, 'r') as f: st.session_state.smart_albums = json.load(f)
    if not st.session_state.captions and os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, 'r') as f: st.session_state.captions = json.load(f)

init_state()

# --- 5. HELPERS ---
def generate_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except: return "Error."

def ask_ai(image_path, question):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(raw_image, question, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except: return "Error."

def detect_faces(image_path):
    img = cv2.imread(image_path)
    if img is None: return 0, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces), faces

# --- 6. STANDARD HELPERS ---
def save_data():
    with open(FAV_PATH, 'w') as f: json.dump(list(st.session_state.favorites), f)
    with open(CAPTIONS_PATH, 'w') as f: json.dump(st.session_state.captions, f)
    with open(ALBUMS_PATH, 'w') as f: json.dump(st.session_state.smart_albums, f)

def toggle_fav(uid):
    if uid in st.session_state.favorites: st.session_state.favorites.remove(uid)
    else: st.session_state.favorites.add(uid)
    save_data()

def copy_path(path):
    if HAS_CLIPBOARD: pyperclip.copy(path); st.toast("Copied!")

def add_to_history(item_id, path):
    st.session_state.history = [h for h in st.session_state.history if h['id'] != item_id]
    st.session_state.history.insert(0, {'id': item_id, 'path': path})
    if len(st.session_state.history) > 20: st.session_state.history.pop()

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("🧠 Neural Gallery")
    st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    
    st.divider()
    active_libs = st.multiselect("Library", options=SEARCH_FOLDERS, default=SEARCH_FOLDERS, format_func=lambda p: FOLDER_MAP.get(p, p))
    year_range = st.slider("Years", 2000, 2026, (2000, 2026))
    
    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.display_items = []
        st.session_state.last_text_query = ""
        st.session_state.viewing_item = None
        st.rerun()

# --- 8. THEATER MODE ---
if st.session_state.viewing_item:
    item_id = st.session_state.viewing_item
    current_list = st.session_state.get('filtered_list', [])
    nav_idx = -1
    for i, item in enumerate(current_list):
        if item['id'] == item_id: nav_idx = i; break
    
    data = collection.get(ids=[item_id], include=['metadatas', 'embeddings'])
    if data and data['metadatas']:
        meta = data['metadatas'][0]
        path = meta.get('path', item_id.split("::")[0])
        ts = meta.get('timestamp', 0)
        is_video = meta.get('media_type') == 'video'
        add_to_history(item_id, path)

        c_main, c_side = st.columns([3, 1])
        
        with c_main:
            n1, n2, n3 = st.columns([1, 4, 1])
            with n1: 
                if nav_idx > 0 and st.button("⬅️ Prev", use_container_width=True):
                    st.session_state.viewing_item = current_list[nav_idx-1]['id']; st.rerun()
            with n2: 
                if st.button("🔙 Grid", use_container_width=True):
                    st.session_state.viewing_item = None; st.rerun()
            with n3: 
                if nav_idx < len(current_list)-1 and st.button("Next ➡️", use_container_width=True):
                    st.session_state.viewing_item = current_list[nav_idx+1]['id']; st.rerun()
            
            if os.path.exists(path):
                if is_video: st.video(path, start_time=int(ts))
                else: st.image(path, use_container_width=True)
            else: st.error("File Lost.")

        with c_side:
            st.subheader("🤖 AI Lab")
            
            # STORYTELLER
            st.caption("Caption")
            if item_id in st.session_state.captions:
                st.markdown(f"<div class='ai-bubble'>📝 {st.session_state.captions[item_id]}</div>", unsafe_allow_html=True)
            else:
                if st.button("📝 Generate Story", use_container_width=True):
                    with st.spinner("Writing..."):
                        cap = generate_caption(path)
                        st.session_state.captions[item_id] = cap
                        save_data(); st.rerun()

            st.divider()
            
            # Q&A
            st.caption("Q&A")
            user_q = st.text_input("Ask question:", placeholder="What color is...?")
            if user_q:
                with st.spinner("Thinking..."):
                    ans = ask_ai(path, user_q)
                    st.markdown(f"<div class='ai-bubble'>💬 <b>{user_q}</b><br>👉 {ans}</div>", unsafe_allow_html=True)

            st.divider()
            
            # FACES
            if not is_video:
                face_count, faces = detect_faces(path)
                st.metric("Faces", face_count)
                if face_count > 0:
                    with st.expander("Show Faces"):
                        img_cv = cv2.imread(path)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        for (x, y, w, h) in faces:
                            st.image(img_cv[y:y+h, x:x+w], width=80)

            st.divider()
            st.code(os.path.basename(path), language="text")
            
            # ACTIONS
            if st.button("🔎 Similar", use_container_width=True):
                if data['embeddings'] is not None and len(data['embeddings']) > 0:
                    res = collection.query(query_embeddings=[data['embeddings'][0]], n_results=50)
                    new_items = []
                    for uid, m in zip(res['ids'][0], res['metadatas'][0]):
                        p = m.get('path', uid.split("::")[0])
                        new_items.append({"id": uid, "path": p, "is_video": '::' in uid, "ts": 0, "date": 0})
                    st.session_state.display_items = new_items
                    st.session_state.viewing_item = None; st.rerun()

            is_fav = item_id in st.session_state.favorites
            if st.button("💔 Un-Fav" if is_fav else "❤️ Favorite", use_container_width=True): toggle_fav(item_id); st.rerun()
            if st.button("📋 Copy Path", use_container_width=True): copy_path(path)
    st.stop()

# --- 9. MAIN DASHBOARD ---
st.title("🧠 Neural Gallery")
tab1, tab2, tab3 = st.tabs(["🔎 Search", "🌌 Galaxy", "🎲 Discovery"])

with tab1:
    c1, c2 = st.columns([5, 1])
    with c1:
        query = st.text_input("Search...", value=st.session_state.last_text_query)
    with c2:
        if st.button("💾 Save"):
            if query: st.session_state.smart_albums[query]=query; save_data(); st.toast("Saved!"); st.rerun()

    if st.session_state.smart_albums:
        cols = st.columns(8)
        for i, (k, v) in enumerate(list(st.session_state.smart_albums.items())[:8]):
            if cols[i].button(k, key=f"alb_{i}"): st.session_state.last_text_query=v; st.rerun()

    if query and (query != st.session_state.last_text_query or not st.session_state.display_items):
        st.session_state.on_this_day_mode = False
        with torch.no_grad():
            text = clip.tokenize([query]).to(device)
            text_features = clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            results = collection.query(query_embeddings=text_features.cpu().numpy().tolist(), n_results=60)
            
            new_items = []
            if results['ids']:
                for uid, m in zip(results['ids'][0], results['metadatas'][0]):
                    p = m.get('path', uid.split("::")[0])
                    if any(p.startswith(f) for f in active_libs):
                        new_items.append({
                            "id": uid, "path": p, "is_video": '::' in uid, 
                            "ts": 0, "date": os.path.getmtime(p) if os.path.exists(p) else 0
                        })
            
            st.session_state.display_items = new_items
            st.session_state.last_text_query = query
            st.rerun()

with tab2:
    st.subheader("🌌 Semantic Galaxy")
    if not HAS_SKLEARN:
        st.error("Please run: pip install scikit-learn")
    else:
        if st.button("🚀 Launch Galaxy Map (Scans 500 random items)"):
            with st.spinner("Mapping the universe..."):
                all_d = collection.get(include=['embeddings', 'metadatas'])
                if all_d['ids'] and len(all_d['ids']) > 50:
                    # Sample
                    idxs = random.sample(range(len(all_d['ids'])), min(500, len(all_d['ids'])))
                    embeddings = []
                    metadata_list = []
                    
                    for i in idxs:
                        if all_d['embeddings'][i] is not None:
                            embeddings.append(all_d['embeddings'][i])
                            metadata_list.append(all_d['metadatas'][i])
                    
                    # PCA Reduction
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(embeddings)
                    
                    # DataFrame
                    df_map = pd.DataFrame(coords, columns=['x', 'y'])
                    # Add path info for tooltip
                    df_map['path'] = [os.path.basename(m.get('path', 'Img')) for m in metadata_list]
                    
                    st.scatter_chart(df_map, x='x', y='y', color='#FF4B4B', size=20)
                    st.success("Similar images are grouped together!")

with tab3:
    if st.button("🎲 Surprise Me"):
        all_d = collection.get(include=['metadatas', 'embeddings'])
        if all_d['ids']:
            idxs = random.sample(range(len(all_d['ids'])), 20)
            ni = []
            for i in idxs:
                uid = all_d['ids'][i]; m = all_d['metadatas'][i]; p = m.get('path', uid.split("::")[0])
                ni.append({"id": uid, "path": p, "is_video": '::' in uid, "ts": 0, "date": 0})
            st.session_state.display_items = ni; st.session_state.last_text_query = ""; st.rerun()

# RENDER
filtered = st.session_state.display_items
st.session_state.filtered_list = filtered

if not filtered and not query: st.info("Ready.")
elif not filtered: st.warning("No results.")
else:
    cols = st.columns(4)
    for i, item in enumerate(filtered):
        with cols[i % 4]:
            path = item['path']
            
            if os.path.exists(path):
                if item['is_video']: st.video(path)
                else: 
                    try: 
                        img = Image.open(path); img.thumbnail((300,300)); st.image(img, use_container_width=True) 
                    except: pass
                
                if st.button(f"📂 {os.path.basename(path)[:15]}", key=f"btn_{i}"):
                    st.session_state.viewing_item = item['id']; st.rerun()