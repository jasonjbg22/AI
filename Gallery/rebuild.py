import chromadb
import clip
import torch
import os
import shutil
from PIL import Image
from tqdm import tqdm

# --- IMPORT CONFIG ---
try:
    import config
except ImportError:
    print("❌ Error: config.py not found. Please create it.")
    exit()

def rebuild():
    print("🛑 STOPPING: Make sure app.py is NOT running.")
    
    # 1. DELETE OLD DB
    if os.path.exists(config.DB_PATH):
        print(f"🗑️  Deleting corrupt database at {config.DB_PATH}...")
        try:
            shutil.rmtree(config.DB_PATH)
            print("✅ Old database removed.")
        except Exception as e:
            print(f"❌ Error deleting DB: {e}")
            return

    # 2. SETUP AI
    print("🧠 Loading AI Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config.MODEL_NAME, device=device)
    
    # 3. SETUP DB
    client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = client.create_collection(name="my_images")
    
    # 4. SCAN FILES
    print("📂 Scanning folders...")
    image_paths = []
    # Use paths from config.py
    for root_path in config.FOLDER_MAP.keys():
        if not os.path.exists(root_path):
            print(f"⚠️ Warning: Folder not found: {root_path}")
            continue
            
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images. Starting processing...")

    # 5. BATCH PROCESS
    BATCH_SIZE = 64
    batch_paths = []
    batch_ids = []
    
    for i, path in enumerate(tqdm(image_paths)):
        try:
            unique_id = f"{path}::{os.path.getmtime(path)}"
            batch_paths.append(path)
            batch_ids.append(unique_id)
            
            if len(batch_paths) >= BATCH_SIZE:
                process_batch(collection, model, preprocess, device, batch_paths, batch_ids)
                batch_paths = []
                batch_ids = []
                
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if batch_paths:
        process_batch(collection, model, preprocess, device, batch_paths, batch_ids)

    print("\n✅ DONE! You can now run 'streamlit run app.py'")

def process_batch(collection, model, preprocess, device, paths, ids):
    images = []
    valid_ids = []
    metadatas = []
    
    for p, uid in zip(paths, ids):
        try:
            image = preprocess(Image.open(p)).unsqueeze(0)
            images.append(image)
            valid_ids.append(uid)
            metadatas.append({
                "path": p,
                "timestamp": os.path.getmtime(p),
                "media_type": "image"
            })
        except:
            continue
            
    if not images: return

    with torch.no_grad():
        image_input = torch.cat(images).to(device)
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
    collection.add(
        embeddings=image_features.cpu().numpy().tolist(),
        metadatas=metadatas,
        ids=valid_ids
    )

if __name__ == "__main__":
    rebuild()