import streamlit as st
import chromadb
import clip
import torch
import os
import time
import json
import random
import hashlib
import threading
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2

# --- 1. CONFIGURATION & CSS ---
st.set_page_config(layout="wide", page_title="Neural Gallery Ultimate", page_icon="⚡")

try:
    import config
except ImportError:
    st.error("⚠️ Critical Error: 'config.py' not found.")
    st.stop()

THUMB_CACHE_DIR = "neural_thumbs"
if not os.path.exists(THUMB_CACHE_DIR):
    os.makedirs(THUMB_CACHE_DIR)

# --- REFINED CSS FOR OVERLAYS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* The Container for Thumbnail + Play Button Overlay */
    .video-container {
        position: relative;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 5px;
    }

    /* Target the Play Button specifically via its wrapper class */
    .play-overlay-wrapper div[data-testid="stButton"] button {
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: 70px !important;
        height: 70px !important;
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: white !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        font-size: 30px !important;
        z-index: 100 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }
    
    .play-overlay-wrapper div[data-testid="stButton"] button:hover {
        background-color: rgba(255, 75, 75, 0.8) !important;
        transform: translate(-50%, -50%) scale(1.1) !important;
    }

    /* Standard Grid Button Styling */
    div[data-testid="stButton"] button {
        border-radius: 10px;
        border: 1px solid #444;
        width: 100%;
        height: 3em;
        font-weight: bold;
    }

    .stVideo { border-radius: 12px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. RESOURCE LOADERS ---
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config.MODEL_NAME, device=device)
    
    taxonomies = {
        "Actions": ["Swimming", "Hiking", "Eating", "Sleeping", "Working", "Driving", "Dancing"],
        "Scenes": ["Beach", "Mountain", "City", "Bedroom", "Kitchen", "Office", "Car", "Forest"],
        "Clothing": ["Casual", "Suit", "Dress", "Swimwear", "Gymwear", "Jacket", "Uniform", "Hat"],
        "People": ["Selfie", "Group", "Portrait", "Baby", "Kids", "Family", "Couple"],
        "Colors": ["Red", "Blue", "Green", "Yellow", "Black", "White", "Orange", "Pink"]
    }
    
    cat_vectors = {}
    with torch.no_grad():
        for cat_type, labels in taxonomies.items():
            prompts = [f"a photo of {c.lower()}" for c in labels]
            tok = clip.tokenize(prompts).to(device)
            vecs = model.encode_text(tok)
            vecs /= vecs.norm(dim=-1, keepdim=True)
            cat_vectors[cat_type] = {"vectors": vecs.type(torch.float16), "labels": labels}
    return model, preprocess, device, cat_vectors

@st.cache_resource
def load_db_data():
    client = chromadb.PersistentClient(path=config.DB_PATH)
    col = client.get_or_create_collection(name="my_images")
    data = col.get(include=['embeddings', 'metadatas'], limit=None)
    
    # Fix for truth-value ambiguity
    if data['embeddings'] is None or len(data['embeddings']) == 0:
        return None, None, None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.tensor(data['embeddings'], dtype=torch.float16).to(device)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    id_map = {uid: i for i, uid in enumerate(data['ids'])}
    return col, embeddings, data['metadatas'], data['ids'], id_map

@st.cache_resource
def classify_all_images(_gpu_index, _taxonomy_data):
    if _gpu_index is None: return {}
    results = {}
    with torch.no_grad():
        for cat_key, cat_data in _taxonomy_data.items():
            scores = _gpu_index @ cat_data["vectors"].T
            results[cat_key] = scores.argmax(dim=1)
    return results

@st.cache_data
def get_counts_fast(_assignments_cpu_numpy, active_libs_tuple, labels, _all_meta):
    if active_libs_tuple:
        valid_mask = [any(os.path.normpath(m.get('path', "")).lower().startswith(lib) for lib in active_libs_tuple) for m in _all_meta]
        filtered_assignments = _assignments_cpu_numpy[np.array(valid_mask)]
    else: filtered_assignments = _assignments_cpu_numpy
    counts = pd.Series(filtered_assignments).value_counts()
    return {f"{name} ({counts.get(i,0)})": name for i, name in enumerate(labels)}

# --- 3. MEDIA HANDLING ---
def load_single_image(item):
    try:
        path, uid = item['path'], item['id']
        path_hash = hashlib.md5(uid.encode('utf-8')).hexdigest()
        cache_path = os.path.join(THUMB_CACHE_DIR, f"{path_hash}.jpg")
        if os.path.exists(cache_path): return Image.open(cache_path)
        if not os.path.exists(path): return None
        if '::' in uid: 
            ts = int(uid.split('::')[1])
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, ts)
            ret, frame = cap.read()
            cap.release()
            if not ret: return None
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else: img = Image.open(path).convert("RGB")
        img.thumbnail((350, 350))
        img.save(cache_path, "JPEG", quality=80)
        return img
    except: return None

def load_batch_parallel(item_list):
    with ThreadPoolExecutor(max_workers=12) as executor:
        return list(executor.map(load_single_image, item_list))

# --- 4. INITIALIZATION ---
try:
    clip_model, clip_prep, device, taxonomy_data = load_models()
    collection, gpu_index, gpu_meta, gpu_ids, gpu_map = load_db_data()
    cached_assignments = classify_all_images(gpu_index, taxonomy_data)
except Exception as e:
    st.error(f"System Error: {e}"); st.stop()

if 'init' not in st.session_state:
    st.session_state.update({
        'init': True, 'display_items': [], 'viewing_item': None, 
        'last_text_query': "", 'favorites': set(), 'unhidden': set(), 
        'page_size': 40, 'last_filter_hash': "", 'playing_in_grid': set(),
        'target_id': None 
    })

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚡ Neural Gallery")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.caption(f"Device: {gpu_name} | {len(gpu_ids) if gpu_ids else 0:,} Items")
    
    active_libs = [os.path.normpath(p).lower() for p, n in config.FOLDER_MAP.items() if st.toggle(n, value=True, key=f"lib_{n}")]
    
    st.divider()
    st.subheader("🎬 Media Filter")
    video_only = st.toggle("🎥 Videos Only", value=False) 
    
    st.divider()
    combined_mask = None
    current_filter_selections = []
    if gpu_index is not None:
        for cat_key, cat_data in taxonomy_data.items():
            with st.expander(cat_key, expanded=False):
                opts_map = get_counts_fast(cached_assignments[cat_key].cpu().numpy(), tuple(active_libs), cat_data["labels"], gpu_meta)
                selected = st.pills("Filter:", list(opts_map.keys()), selection_mode="multi", key=f"p_{cat_key}")
                if selected:
                    current_filter_selections.append(tuple(selected))
                    cat_mask = torch.zeros(len(gpu_ids), dtype=torch.bool, device=device)
                    for lbl in selected:
                        cat_mask |= (cached_assignments[cat_key] == cat_data["labels"].index(opts_map[lbl]))
                    combined_mask = cat_mask if combined_mask is None else combined_mask & cat_mask

    filter_hash = f"{hash(tuple(current_filter_selections))}_{video_only}_{hash(tuple(active_libs))}"
    filters_changed = filter_hash != st.session_state.last_filter_hash
    st.session_state.last_filter_hash = filter_hash

    if st.button("🔄 Reset View", use_container_width=True):
        st.session_state.update({'display_items': [], 'last_text_query': "", 'viewing_item': None, 'unhidden': set(), 'page_size': 40, 'playing_in_grid': set(), 'target_id': None})
        st.rerun()

# --- 6. THEATER MODE (Full View) ---
if st.session_state.viewing_item:
    it_id = st.session_state.viewing_item
    data = collection.get(ids=[it_id], include=['metadatas', 'embeddings'])
    if data['metadatas']:
        path, ts = data['metadatas'][0].get('path', ""), data['metadatas'][0].get('timestamp', 0)
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("⬅️ Back to Gallery"): st.session_state.viewing_item = None; st.rerun()
            if os.path.exists(path):
                if '::' in it_id: st.video(path, start_time=int(ts), autoplay=True)
                else: st.image(path, use_container_width=True)
        with c2:
            st.subheader("Actions")
            if st.button("✨ Find Similar", use_container_width=True):
                st.session_state.target_id = it_id
                st.session_state.viewing_item = None; st.rerun()
            if st.button("📂 Open Folder", use_container_width=True): os.startfile(os.path.dirname(path))
    st.stop()

# --- 7. MAIN INTERFACE ---
st.title("📸 Neural Gallery Turbo")
q = st.text_input("Search description...", value=st.session_state.last_text_query)

# SEARCH LOGIC
if (q != st.session_state.last_text_query or filters_changed or st.session_state.target_id or not st.session_state.display_items):
    with torch.no_grad():
        if st.session_state.target_id:
            target_data = collection.get(ids=[st.session_state.target_id], include=['embeddings'])
            vec = torch.tensor(target_data['embeddings'][0], dtype=torch.float16).to(device)
            scores = (gpu_index @ vec.T).squeeze()
            st.session_state.target_id = None 
        elif q:
            vec = clip_model.encode_text(clip.tokenize([q]).to(device))
            vec /= vec.norm(dim=-1, keepdim=True)
            scores = (gpu_index @ vec.T).squeeze()
        else: scores = torch.ones(len(gpu_ids), device=device)

        if combined_mask is not None: scores = scores.masked_fill(~combined_mask, -1.0)
        for idx, meta in enumerate(gpu_meta):
            p_norm = os.path.normpath(meta.get('path', "")).lower()
            if not (any(p_norm.startswith(lib) for lib in active_libs) and (not video_only or meta.get('media_type') == 'video')):
                scores[idx] = -1.0
        
        _, indices = scores.topk(min(st.session_state.page_size, len(gpu_ids)))
        st.session_state.display_items = [{"id": gpu_ids[idx], "path": gpu_meta[idx]['path']} for idx in indices.cpu().numpy() if scores[idx] > -0.5]
        st.session_state.last_text_query = q; st.rerun()

# --- 8. RENDER GRID ---
filtered = st.session_state.display_items
st.session_state.filtered_list = filtered

if filtered:
    loaded_imgs = load_batch_parallel(filtered)
    cols = st.columns(4)
    for i, item in enumerate(filtered):
        with cols[i % 4]:
            is_video = '::' in item['id']
            
            if is_video and item['id'] in st.session_state.playing_in_grid:
                ts = int(item['id'].split('::')[1])
                st.video(item['path'], start_time=ts, autoplay=True)
                if st.button("⏹ Stop", key=f"stop_{item['id']}"):
                    st.session_state.playing_in_grid.remove(item['id']); st.rerun()
            else:
                # WRAPPER FOR OVERLAY
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                if loaded_imgs[i] is not None:
                    st.image(loaded_imgs[i], use_container_width=True)
                
                if is_video:
                    # Target the button with a special HTML wrapper for the CSS
                    st.markdown('<div class="play-overlay-wrapper">', unsafe_allow_html=True)
                    if st.button("▶", key=f"play_{item['id']}"):
                        st.session_state.playing_in_grid.add(item['id']); st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # FOOTER BUTTONS
            c1, c2 = st.columns(2)
            if c1.button("🔍 View", key=f"full_{item['id']}"):
                st.session_state.viewing_item = item['id']; st.rerun()
            if c2.button("✨ Sim", key=f"sim_{item['id']}"):
                st.session_state.target_id = item['id']; st.rerun()

    if st.button("⬇️ Load More"): st.session_state.page_size += 40; st.rerun()