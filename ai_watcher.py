# =============================
# CONFIGURATION
# =============================
PROJECT_ROOT = os.getcwd()
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backup")
CONTEXT_FILE = os.path.join(PROJECT_ROOT, "_ai_context.txt")
GITIGNORE_FILE = os.path.join(PROJECT_ROOT, ".gitignore")

# LIMITS
MAX_FILE_SIZE = 100 * 1024  # 100 KB limit per file (prevents massive context)

# IGNORE SETTINGS
IGNORE_DIRS = {
    ".git", ".idea", "__pycache__", "node_modules", "backup", "venv", "env", 
    "bin", "obj", "lib", ".vscode", "dist", "build", "coverage", ".next", 
    "target", "out", "Gallery", "gallery"  # Hardcoded Gallery exclusion
}

IGNORE_EXTS = {
    # Binaries/Executables
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pkl", ".pyc", ".pyd",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp", ".tiff", ".webp",
    # Audio/Video
    ".mp3", ".wav", ".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm",
    # Archives
    ".zip", ".tar", ".gz", ".7z", ".rar", ".jar", ".war",
    # Misc
    ".bak", ".log", ".map", ".lock", ".pdf", ".db", ".sqlite", ".sqlite3",
    ".arw"  # Added RAW photo extension since you use them
}
