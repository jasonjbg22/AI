import os
import re
import time
import shutil
from datetime import datetime
import pyperclip

# =============================
# CONFIGURATION
# =============================
PROJECT_ROOT = os.getcwd()
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backup")
CONTEXT_FILE = os.path.join(PROJECT_ROOT, "_ai_context.txt")
VERSION_LOG = os.path.join(PROJECT_ROOT, "_ai_version_log.txt")

IGNORE_DIRS = {".git", ".idea", "__pycache__", "node_modules", "backup", "venv", "env", "bin", "obj", "lib", ".vscode"}
IGNORE_EXTS = {".exe", ".dll", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".bak", ".svg", ".pdf"}
IGNORE_FILES = {"ai_watcher.py", "_ai_context.txt", "_ai_version_log.txt", "package-lock.json"}

os.makedirs(BACKUP_DIR, exist_ok=True)
last_clipboard = ""
last_project_mtime = 0

print("=== AI AUTO-SYNC ENGINE ===")
print(f"Watching: {PROJECT_ROOT}")
print("Mode: Clean Deploy Engine\n")

# =============================
# CONTEXT BUILDER
# =============================

def get_project_mtime():
    max_mtime = 0
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f == os.path.basename(CONTEXT_FILE):
                continue
            try:
                full_path = os.path.join(root, f)
                mtime = os.stat(full_path).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
            except:
                pass
    return max_mtime


def generate_tree(startpath):
    tree = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f in IGNORE_FILES or any(f.endswith(ext) for ext in IGNORE_EXTS):
                continue
            tree.append(f"{subindent}{f}")
    return "\n".join(tree)


def rebuild_context_file():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rebuilding context...")

    content_blocks = []
    content_blocks.append("=== PROJECT MAP ===\n" + generate_tree(PROJECT_ROOT) + "\n=== END MAP ===")

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f in IGNORE_FILES or any(f.endswith(ext) for ext in IGNORE_EXTS):
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as file_obj:
                    file_content = file_obj.read()
                    content_blocks.append(
                        f"=== FILE: {rel_path} ===\n{file_content}\n=== END FILE ==="
                    )
            except Exception as e:
                content_blocks.append(
                    f"=== FILE: {rel_path} ===\n[ERROR: {e}]\n=== END FILE ==="
                )

    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(content_blocks))

    print("Context updated.")


# =============================
# DEPLOY ENGINE (CLEAN WRITE)
# =============================

def backup_file(path):
    if not os.path.exists(path):
        return

    filename = os.path.basename(path)
    file_backup_dir = os.path.join(BACKUP_DIR, filename)
    os.makedirs(file_backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    shutil.copy2(path, os.path.join(file_backup_dir, f"{timestamp}.bak"))

    backups = sorted(os.listdir(file_backup_dir))
    while len(backups) > 10:
        os.remove(os.path.join(file_backup_dir, backups.pop(0)))


def extract_val(tag, text):
    m = re.search(rf"^\s*{tag}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def process_deploy(text):
    blocks = re.split(r"(?m)(?=^@AI-FILE:)", text)
    changes = {}
    deletes = set()

    try:
        for b in blocks:
            if not b.strip().startswith("@AI-FILE:"):
                continue

            rel_path = extract_val("@AI-FILE", b)
            path = os.path.join(PROJECT_ROOT, rel_path)

            if extract_val("@AI-ACTION", b) == "DELETE":
                deletes.add(path)
                continue

            file_content = ""

            for section_block in re.split(r"(?=@AI-SECTION:)", b):
                if "@AI-SECTION:" not in section_block:
                    continue

                lines = section_block.splitlines()

                code_lines = [
                    l for l in lines
                    if not l.strip().startswith("@AI-")
                ]

                clean_code = "\n".join(code_lines).strip()

                if clean_code:
                    file_content += clean_code + "\n\n"

            changes[path] = file_content.strip() + "\n"

    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("[COMMITTING CLEAN FILES]")

    for p in deletes:
        if os.path.exists(p):
            os.remove(p)

    for p, c in changes.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        backup_file(p)
        with open(p, "w", encoding="utf-8") as f:
            f.write(c)

    print("=== DEPLOY SUCCESS (CLEAN WRITE) ===")

    global last_project_mtime
    last_project_mtime = 0


# =============================
# MAIN LOOP
# =============================

while True:
    try:
        clip = pyperclip.paste()

        if clip != last_clipboard:
            last_clipboard = clip
            if "@AI-FILE:" in clip:
                print("\n=== DEPLOY DETECTED ===")
                process_deploy(clip)

        current_mtime = get_project_mtime()
        if current_mtime > last_project_mtime:
            last_project_mtime = current_mtime
            rebuild_context_file()

        time.sleep(1)

    except KeyboardInterrupt:
        break
