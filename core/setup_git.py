import os
import subprocess

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        return False

def setup_git():
    print("Initializing Git Repository...")
    
    # 1. Init Git
    if not os.path.exists(".git"):
        run_command("git init")
    
    # 2. Basic Configuration
    run_command('git config core.autocrlf true')
    
    # 3. Initial Add and Commit
    print("Staging files...")
    run_command("git add .")
    run_command('git commit -m "Initial AI Engine Setup with Core structure"')
    
    print("\nNext Steps:")
    print("1. Create a PRIVATE repo on GitHub.")
    print("2. Run: git remote add origin <your_url>")
    print("3. Run: git push -u origin main")

if __name__ == "__main__":
    # Ensure we are in the root when running
    os.chdir("..")
    setup_git()
