import os
import clip
import torch
import chromadb
import cv2
import time
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- MONITORING TOOLS ---
try:
    import psutil
    import GPUtil
    MONITORING = True
except ImportError:
    MONITORING = False

# --- CONFIGURATION ---
SEARCH_FOLDERS = [
    r"G:\AllPhotosDatabase\AI_ARt\em",
    r"G:\AllPhotosDatabase\2em",
    r"G:\AllPhotosDatabase\Photos",
]

DB_PATH = "./my_image_db"
MODEL_NAME = "ViT-L/14"
BATCH_SIZE = 64           # Increased slightly for better throughput
NUM_WORKERS = 4           # Keep at 4 for HDD stability

# VIDEO SETTINGS
FRAME_INTERVAL = 20
# ---------------------

Image.MAX_IMAGE_PIXELS = None

class MediaDataset(Dataset):
    def __init__(self, media_tasks, preprocess):
        self.media_tasks = media_tasks
        self.preprocess = preprocess
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.media_tasks)

    def __getitem__(self, idx):
        file_path, timestamp = self.media_tasks[idx]
        
        try:
            image = None
            
            # CASE A: Video Task
            if timestamp is not None:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened(): return None, None, None, None
                
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = cap.read()
                cap.release()
                
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)

            # CASE B: Image Task
            else:
                image = Image.open(file_path)
                
                # --- HDD OPTIMIZATION: DRAFT LOADING ---
                # If it's a JPEG, we ask the drive to only send us a smaller version.
                # This drastically reduces the data read from the disk.
                if image.format == 'JPEG':
                    image.draft('RGB', (512, 512))
                # ---------------------------------------

                if image.mode != 'RGB':
                    image = image.convert('RGB')

            if image:
                processed_image = self.preprocess(image)
                unique_id = file_path if timestamp is None else f"{file_path}::{timestamp}"
                return processed_image, unique_id, file_path, timestamp
            
            return None, None, None, None

        except Exception:
            return None, None, None, None

def collate_fn_safe(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return None, None, None, None
    images, ids, paths, timestamps = zip(*batch)
    return torch.stack(images), ids, paths, timestamps

def get_system_stats():
    stats = {}
    if MONITORING:
        stats["CPU"] = f"{psutil.cpu_percent()}%"
        mem = psutil.virtual_memory()
        stats["RAM"] = f"{mem.percent}%"
        gpus = GPUtil.getGPUs()
        if gpus:
            stats["GPU"] = f"{int(gpus[0].load * 100)}%"
            stats["VRAM"] = f"{int(gpus[0].memoryUtil * 100)}%"
    return stats

def main():
    torch.multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 💾 HDD OPTIMIZED SYNC STARTED ON: {device.upper()} ---")
    
    print("Loading AI Model...")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="my_images")

    print("📂 Scanning file system...")
    tasks = [] 
    files_on_disk = set()

    # We want to preserve the OS order (likely Folder order) to minimize seeking
    for folder in SEARCH_FOLDERS:
        if os.path.exists(folder):
            print(f"   - Scanning: {folder}")
            for root, dirs, files in os.walk(folder):
                for file in files:
                    path = os.path.join(root, file)
                    lower = file.lower()
                    files_on_disk.add(path)

                    if lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                        tasks.append((path, None))
                    elif lower.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                        try:
                            # Minimal open just to check duration
                            cap = cv2.VideoCapture(path)
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                duration = frames / fps if fps > 0 else 0
                                cap.release()
                                if duration > 0:
                                    for t in range(0, int(duration), FRAME_INTERVAL):
                                        tasks.append((path, t))
                        except: pass
        else:
            print(f"⚠️ Warning: Folder not found: {folder}")

    # PRUNING
    print("\n🧹 Pruning deleted files...")
    existing_ids = collection.get()['ids']
    ids_to_delete = []
    
    for db_id in existing_ids:
        original_path = db_id.split("::")[0]
        is_watched = any(original_path.startswith(f) for f in SEARCH_FOLDERS)
        if is_watched and original_path not in files_on_disk:
            ids_to_delete.append(db_id)
            
    if ids_to_delete:
        print(f"   - Removing {len(ids_to_delete)} dead links...")
        chunk_size = 1000
        for i in range(0, len(ids_to_delete), chunk_size):
            collection.delete(ids=ids_to_delete[i:i+chunk_size])
    else:
        print("   - No dead links found.")

    # ADDING
    print("\n🔍 Checking for new files...")
    existing_ids_set = set(collection.get()['ids'])
    new_tasks = []
    
    for path, ts in tasks:
        check_id = path if ts is None else f"{path}::{ts}"
        if check_id not in existing_ids_set:
            new_tasks.append((path, ts))

    if not new_tasks:
        print("✅ Library is fully synchronized!")
        return

    # --- HDD OPTIMIZATION: DO NOT SORT ---
    # We purposefully removed the "sort by image/video" line.
    # Now the script processes files in the exact order they sit in the folders.
    # This keeps the hard drive head moving forward smoothly.
    
    print(f"⚡ Indexing {len(new_tasks)} new items (Sequential Mode)...")

    dataset = MediaDataset(new_tasks, preprocess)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn_safe,
        pin_memory=True
    )

    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Indexing")
        
        for i, batch in enumerate(pbar):
            batch_imgs, batch_ids, batch_paths, batch_ts = batch
            if batch_imgs is None: continue
            
            features = model.encode_image(batch_imgs.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            
            metadatas = []
            for p, t in zip(batch_paths, batch_ts):
                meta = {"path": p}
                if t is not None:
                    meta["timestamp"] = int(t)
                    meta["media_type"] = "video"
                else:
                    meta["media_type"] = "image"
                metadatas.append(meta)

            collection.add(
                ids=list(batch_ids),
                embeddings=features.cpu().numpy().tolist(),
                metadatas=metadatas
            )

            if i % 5 == 0:
                stats = get_system_stats()
                pbar.set_postfix(stats)

    print("\n--- ✅ Sync Complete! ---\n")

if __name__ == "__main__":
    main()