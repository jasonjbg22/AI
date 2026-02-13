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
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- LIBRARY CHECKS ---
try:
    from sklearn.cluster import KMeans, DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Neural Gallery Ultimate", page_icon="🧠")

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

# --- 3. RESOURCES ---
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Search (CLIP)
    model, preprocess = clip.load(MODEL_NAME, device=device)
    # Caption (BLIP)
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    gen_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    return model, preprocess, proc, gen_model, device

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name="my_images")

try:
    clip_model, clip_prep, blip_proc, blip_model, device = load_models()
    collection = load_db()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- 4. STATE ---
def init_state():
    defaults = {
        'display_items': [], 'viewing_item': None, 'last_text_query': "", 
        'sort_order': "AI Similarity", 'on_this_day_mode': False,
        'history': [], 'favorites': set(), 'smart_albums': {}, 'captions': {},
        'face_clusters': {}, 'trip_clusters': {}, 'visual_search_img': None
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

# --- 5. LOGIC ---
def save_data():
    with open(FAV_PATH, 'w') as f: json.dump(list(st.session_state.favorites), f)
    with open(CAPTIONS_PATH, 'w') as f: json.dump(st.session_state.captions, f)
    with open(ALBUMS_PATH, 'w') as f: json.dump(st.session_state.smart_albums, f)

def generate_caption(path):
    try:
        raw = Image.open(path).convert('RGB')
        ins = blip_proc(raw, return_tensors="pt").to(device)
        out = blip_model.generate(**ins)
        return blip_proc.decode(out[0], skip_special_tokens=True)
    except: return "Error"

def ask_ai(path, q):
    try:
        raw = Image.open(path).convert('RGB')
        ins = blip_proc(raw, q, return_tensors="pt").to(device)
        out = blip_model.generate(**ins)
        return blip_proc.decode(out[0], skip_special_tokens=True)
    except: return "Error"

def get_faces(path):
    img = cv2.imread(path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rects = cascade.detectMultiScale(gray, 1.1, 4)
    faces = []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in rects:
        face_crop = Image.fromarray(img_rgb[y:y+h, x:x+w])
        faces.append(face_crop)
    return faces

def get_exif_gps(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif: return None
        gps_data = exif.get(34853) # 34853 is GPSInfo
        if not gps_data: return None
        
        def to_deg(v): return float(v[0]) + (float(v[1])/60.0) + (float(v[2])/3600.0)
        
        if 2 in gps_data and 4 in gps_data:
            lat = to_deg(gps_data[2])
            lon = to_deg(gps_data[4])
            if gps_data[1] == 'S': lat = -lat
            if gps_data[3] == 'W': lon = -lon
            return lat, lon
    except: return None
    return None

def auto_fix_image(path):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img = cv2.imread(path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def toggle_fav(uid):
    if uid in st.session_state.favorites: st.session_state.favorites.remove(uid)
    else: st.session_state.favorites.add(uid)
    save_data()

def copy_path(path):
    if HAS_CLIPBOARD: pyperclip.copy(path); st.toast("Copied!")

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("🧠 Neural Gallery")
    st.caption(f"Hardware: {torch.cuda.get_device_name(0)}")
    
    st.divider()
    st.write("🖼️ **Visual Reverse Search**")
    uploaded_file = st.file_uploader("Drop an image here to find matches", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        st.session_state.visual_search_img = Image.open(uploaded_file)
        if st.button("🚀 Search Image"):
            with torch.no_grad():
                img = clip_prep(st.session_state.visual_search_img).unsqueeze(0).to(device)
                vec = clip_model.encode_image(img)
                vec /= vec.norm(dim=-1, keepdim=True)
                res = collection.query(query_embeddings=vec.cpu().numpy().tolist(), n_results=60)
                ni = []
                for uid, m in zip(res['ids'][0], res['metadatas'][0]):
                    p = m.get('path', uid.split("::")[0])
                    ni.append({"id": uid, "path": p, "is_video": '::' in uid, "ts": 0, "date": 0})
                st.session_state.display_items = ni
                st.session_state.last_text_query = "Visual Search"
                st.rerun()

    st.divider()
    active_libs = st.multiselect("Sources", options=SEARCH_FOLDERS, default=SEARCH_FOLDERS, format_func=lambda p: FOLDER_MAP.get(p, p))
    year_range = st.slider("Years", 2000, 2026, (2000, 2026))
    
    st.divider()
    if st.button("🔄 Reset View"):
        st.session_state.display_items = []
        st.session_state.last_text_query = ""
        st.session_state.viewing_item = None
        st.rerun()

# --- 7. THEATER MODE ---
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
        
        c1, c2 = st.columns([3, 1])
        with c1:
            n1, n2, n3 = st.columns([1, 4, 1])
            with n1: 
                if nav_idx > 0 and st.button("⬅️ Prev"):
                    st.session_state.viewing_item = current_list[nav_idx-1]['id']; st.rerun()
            with n2: 
                if st.button("🔙 Grid"):
                    st.session_state.viewing_item = None; st.rerun()
            with n3: 
                if nav_idx < len(current_list)-1 and st.button("Next ➡️"):
                    st.session_state.viewing_item = current_list[nav_idx+1]['id']; st.rerun()
            
            if os.path.exists(path):
                if is_video: st.video(path, start_time=int(ts))
                else: 
                    # Auto Fix Toggle
                    if st.toggle("✨ Auto-Enhance Preview"):
                        fixed = auto_fix_image(path)
                        st.image(fixed, use_container_width=True, caption="AI Enhanced Preview")
                    else:
                        st.image(path, use_container_width=True)
            else: st.error("File missing")

        with c2:
            st.subheader("🤖 AI Analysis")
            
            if item_id in st.session_state.captions:
                st.markdown(f"<div class='ai-bubble'>📝 {st.session_state.captions[item_id]}</div>", unsafe_allow_html=True)
            else:
                if st.button("📝 Write Story"):
                    with st.spinner("Dreaming..."):
                        st.session_state.captions[item_id] = generate_caption(path)
                        save_data(); st.rerun()
            
            q = st.text_input("Ask the photo:", placeholder="What is happening?")
            if q:
                with st.spinner("Thinking..."):
                    ans = ask_ai(path, q)
                    st.markdown(f"<div class='ai-bubble'>💬 {ans}</div>", unsafe_allow_html=True)
            
            st.divider()
            if st.button("🔎 Find Similar"):
                if data['embeddings'] and len(data['embeddings']) > 0:
                    res = collection.query(query_embeddings=[data['embeddings'][0]], n_results=50)
                    ni = []
                    for uid, m in zip(res['ids'][0], res['metadatas'][0]):
                        p = m.get('path', uid.split("::")[0])
                        ni.append({"id": uid, "path": p, "is_video": '::' in uid, "ts": 0, "date": 0})
                    st.session_state.display_items = ni; st.session_state.viewing_item = None; st.rerun()
            
            is_fav = item_id in st.session_state.favorites
            if st.button("💔 Un-Fav" if is_fav else "❤️ Favorite"): toggle_fav(item_id); st.rerun()
            if st.button("📋 Copy Path"): copy_path(path)
    st.stop()

# --- 8. DASHBOARD ---
st.title("🧠 Neural Gallery")
t1, t2, t3, t4, t5 = st.tabs(["🔎 Search", "👥 Identities", "🗺️ Trips", "📅 Life-Log", "🎲 Discovery"])

# TAB 1: SEARCH
with t1:
    c1, c2 = st.columns([5, 1])
    with c1: q = st.text_input("Search...", value=st.session_state.last_text_query)
    with c2: 
        if st.button("💾 Save"): st.session_state.smart_albums[q]=q; save_data(); st.rerun()
    
    if st.session_state.smart_albums:
        cols = st.columns(8)
        for i, (k,v) in enumerate(list(st.session_state.smart_albums.items())[:8]):
            if cols[i].button(k, key=f"alb_{i}"): st.session_state.last_text_query=v; st.rerun()

    if q and (q != st.session_state.last_text_query or not st.session_state.display_items):
        with torch.no_grad():
            tok = clip.tokenize([q]).to(device)
            vec = clip_model.encode_text(tok)
            vec /= vec.norm(dim=-1, keepdim=True)
            res = collection.query(query_embeddings=vec.cpu().numpy().tolist(), n_results=60)
            ni = []
            if res['ids']:
                for uid, m in zip(res['ids'][0], res['metadatas'][0]):
                    p = m.get('path', uid.split("::")[0])
                    if any(p.startswith(f) for f in active_libs):
                        ni.append({"id": uid, "path": p, "is_video": '::' in uid, "ts": 0, "date": os.path.getmtime(p) if os.path.exists(p) else 0})
            st.session_state.display_items = ni
            st.session_state.last_text_query = q
            st.rerun()

# TAB 2: IDENTITIES
with t2:
    st.subheader("👥 Face Clustering")
    if not HAS_SKLEARN: st.error("Missing sklearn")
    else:
        if st.button("🚀 Scan & Group Faces"):
            with st.status("Scanning...") as status:
                all_d = collection.get(limit=1000)
                face_emb = []
                face_meta = []
                for i in random.sample(range(len(all_d['ids'])), min(200, len(all_d['ids']))):
                    p = all_d['metadatas'][i].get('path', "")
                    if os.path.exists(p) and p.lower().endswith(('jpg', 'png')):
                        for fimg in get_faces(p):
                            with torch.no_grad():
                                proc = clip_prep(fimg).unsqueeze(0).to(device)
                                emb = clip_model.encode_image(proc)
                                emb /= emb.norm(dim=-1, keepdim=True)
                                face_emb.append(emb.cpu().numpy()[0])
                                face_meta.append({"id": all_d['ids'][i], "path": p})
                
                if len(face_emb) > 5:
                    k = max(3, min(10, len(face_emb)//10))
                    km = KMeans(n_clusters=k)
                    lbls = km.fit_predict(face_emb)
                    clus = {}
                    for idx, l in enumerate(lbls):
                        if l not in clus: clus[l] = []
                        clus[l].append(face_meta[idx])
                    st.session_state.face_clusters = clus
                    status.update(label="Done!", state="complete")

    if st.session_state.face_clusters:
        for cid, items in st.session_state.face_clusters.items():
            with st.expander(f"👤 Person {cid+1} ({len(items)} photos)"):
                cols = st.columns(6)
                for i, item in enumerate(items[:6]):
                    with cols[i]:
                        st.image(item['path'], use_container_width=True)
                        if st.button("View", key=f"f_{cid}_{i}"): st.session_state.viewing_item = item['id']; st.rerun()

# TAB 3: TRIPS
with t3:
    st.subheader("🗺️ Trip Detector")
    if not HAS_SKLEARN: st.error("Missing sklearn")
    else:
        if st.button("🌍 Scan for Trips"):
            with st.spinner("Triangulating..."):
                all_d = collection.get(limit=1000)
                coords = []
                trip_meta = []
                for i in range(len(all_d['ids'])):
                    p = all_d['metadatas'][i].get('path', "")
                    if os.path.exists(p) and p.lower().endswith(('jpg', 'jpeg')):
                        gps = get_exif_gps(p)
                        if gps:
                            coords.append(gps)
                            trip_meta.append({"id": all_d['ids'][i], "path": p})
                
                if coords:
                    # DBSCAN: eps=0.5 (approx 50km), min_samples=5
                    # Groups points within 50km into a "Trip"
                    db = DBSCAN(eps=0.5, min_samples=5).fit(coords)
                    labels = db.labels_
                    t_clus = {}
                    for idx, l in enumerate(labels):
                        if l == -1: continue # Noise
                        if l not in t_clus: t_clus[l] = []
                        t_clus[l].append(trip_meta[idx])
                    st.session_state.trip_clusters = t_clus
        
        if st.session_state.trip_clusters:
            st.success(f"Found {len(st.session_state.trip_clusters)} Trips!")
            for tid, items in st.session_state.trip_clusters.items():
                with st.expander(f"✈️ Trip #{tid+1} ({len(items)} photos)"):
                    cols = st.columns(6)
                    for i, item in enumerate(items[:6]):
                        with cols[i]:
                            st.image(item['path'], use_container_width=True)
                    if st.button(f"View Trip {tid+1}", key=f"trip_{tid}"):
                        st.session_state.display_items = [{"id": x['id'], "path": x['path'], "is_video": False, "ts": 0, "date": 0} for x in items]
                        st.rerun()

# TAB 4: LIFE-LOG
with t4:
    st.subheader("📅 Life-Log Heatmap")
    if not HAS_ALTAIR: st.error("Missing altair")
    else:
        if st.button("🔄 Map"):
            all_d = collection.get()
            dates = []
            for m in all_d['metadatas']:
                p = m.get('path', "")
                if os.path.exists(p):
                    dt = datetime.fromtimestamp(os.path.getmtime(p))
                    dates.append(dt.strftime("%Y-%m-%d"))
            df = pd.DataFrame(dates, columns=['date'])
            df['count'] = 1
            df = df.groupby('date').count().reset_index()
            chart = alt.Chart(df).mark_rect().encode(
                x=alt.X('date:T', timeUnit='yearmonthdate'), y='count:Q', color='count:Q', tooltip=['date', 'count']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

# TAB 5: DISCOVERY
with t5:
    if st.button("🎲 Surprise Me"):
        all_d = collection.get(include=['metadatas'])
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

if not filtered and not q: st.info("Ready.")
elif not filtered: st.warning("No results.")
else:
    cols = st.columns(4)
    for i, item in enumerate(filtered):
        with cols[i % 4]:
            if os.path.exists(item['path']):
                if item['is_video']: st.video(item['path'])
                else: 
                    try: 
                        img = Image.open(item['path']); img.thumbnail((300,300)); st.image(img, use_container_width=True) 
                    except: pass
                
                if st.button(f"📂 {os.path.basename(item['path'])[:15]}", key=f"btn_{i}"):
                    st.session_state.viewing_item = item['id']; st.rerun()