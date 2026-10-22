import subprocess
import sys

def main():
    print("Starting YouTube Shorts Pipeline from GitHub Actions...")
    # Yeh command cron_pipeline.py ko execute karegi --execute flag ke sath
    result = subprocess.run([sys.executable, "cron_pipeline.py", "--execute"])
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
