import os
import subprocess
import sys
import time
import socket

# ==================================================
# PROJECT SETUP
# ==================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WATCHER_PATH = os.path.join(PROJECT_ROOT, "core", "ai_watcher.py")
MACHINE_ID = socket.gethostname().upper()

# ==================================================
# UTIL
# ==================================================
def run_git(command):
    return subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ==================================================
# SAFE PULL LOGIC
# ==================================================
def safe_pull():
    print_section(f"[{MACHINE_ID}] SYNCING WITH CLOUD")

    # Stage everything
    run_git("git add .")

    # Commit if needed
    run_git(f'git commit -m "Auto-save before pull ({MACHINE_ID})"')

    # Pull safely with autostash
    result = run_git("git pull origin main --rebase --autostash")

    if result.returncode != 0:
        print("Pull failed. Attempting rebase abort...")
        run_git("git rebase --abort")
        print(result.stderr)
    else:
        print("✔ Cloud sync successful.")

# ==================================================
# SAFE PUSH LOGIC
# ==================================================
def safe_push():
    print_section(f"[{MACHINE_ID}] PUSHING TO CLOUD")

    run_git("git add .")
    run_git(f'git commit -m "Auto-Sync from {MACHINE_ID} {time.strftime("%H:%M")}"')

    result = run_git("git push origin main")

    if result.returncode != 0:
        print("Push rejected. Pulling latest and retrying...")
        safe_pull()
        result = run_git("git push origin main")

        if result.returncode != 0:
            print("Push still failed.")
            print(result.stderr)
        else:
            print("✔ Push successful after retry.")
    else:
        print("✔ Push successful.")

# ==================================================
# START WATCHER
# ==================================================
def start_watcher():
    if not os.path.exists(WATCHER_PATH):
        print("Watcher not found:", WATCHER_PATH)
        sys.exit(1)

    print_section(f"[{MACHINE_ID}] WATCHER ACTIVE")
    return subprocess.Popen([sys.executable, WATCHER_PATH], cwd=PROJECT_ROOT)

# ==================================================
# MAIN LOOP
# ==================================================
def main():
    safe_pull()
    watcher = start_watcher()

    try:
        while watcher.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping sync...")
        watcher.terminate()
        safe_push()

if __name__ == "__main__":
    main()
