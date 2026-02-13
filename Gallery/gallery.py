import os
import clip
import torch
import chromadb
import cv2
import time
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- IMPORT CONFIG ---
try:
    import config
except ImportError:
    print("❌ config.py not found!")
    exit()

# --- CONFIGURATION FROM CONFIG.PY ---
SEARCH_FOLDERS = list(config.FOLDER_MAP.keys())
DB_PATH = config.DB_PATH
MODEL_NAME = config.MODEL_NAME
BATCH_SIZE = 64           
NUM_WORKERS = 4           
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
            if timestamp is not None:
                cap = cv2.VideoCapture(file_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, timestamp)
                ret, frame = cap.read()
                cap.release()
                if not ret: return None, None, None, None
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                uid = f"{file_path}::{timestamp}"
            else:
                image = Image.open(file_path).convert("RGB")
                uid = file_path
            return self.preprocess(image), uid, file_path, timestamp
        except:
            return None, None, None, None

def collate_fn_safe(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch: return None, None, None, None
    imgs, ids, paths, ts = zip(*batch)
    return torch.stack(imgs), ids, paths, ts

def main():
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="my_images")

    existing_ids = set(collection.get(include=[])['ids'])
    new_tasks = []

    print("📂 Scanning folders...")
    for folder in SEARCH_FOLDERS:
        for root, _, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)
                ext = file.lower()
                if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    if path not in existing_ids:
                        new_tasks.append((path, None))
                elif ext.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    # Check first frame to see if indexed
                    if f"{path}::{FRAME_INTERVAL}" not in existing_ids:
                        cap = cv2.VideoCapture(path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        for f in range(FRAME_INTERVAL, frame_count, FRAME_INTERVAL * 50):
                            new_tasks.append((path, f))

    if not new_tasks:
        print("✅ Everything up to date.")
        return

    print(f"⚡ Indexing {len(new_tasks)} items...")
    dataset = MediaDataset(new_tasks, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn_safe)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, ids, paths, ts = batch
            if imgs is None: continue
            features = model.encode_image(imgs.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            metadatas = [{"path": p, "media_type": "video" if t else "image", "timestamp": int(t) if t else 0} for p, t in zip(paths, ts)]
            collection.add(ids=list(ids), embeddings=features.cpu().numpy().tolist(), metadatas=metadatas)

if __name__ == "__main__":
    main()