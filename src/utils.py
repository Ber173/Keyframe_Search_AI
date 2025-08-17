import os

def ensure_dirs():
    os.makedirs("data/videos", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
