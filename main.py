import os
import cv2
import torch
import clip
import numpy as np
import shutil
import argparse
from PIL import Image
from numpy.linalg import norm

# ----------------------------
# 1. Parse arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="Path to input video")
parser.add_argument("--query", type=str, required=True, help="Text query to search in video")
parser.add_argument("--topk", type=int, default=5, help="Number of keyframes to save")
parser.add_argument("--frame_step", type=int, default=30, help="Take 1 frame every N frames (default=30 ~1s if 30fps)")
args = parser.parse_args()

# ----------------------------
# 2. Load CLIP model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------
# Helper functions
# ----------------------------
def is_blurry(image, threshold=100.0):
    """Check if image is blurry using variance of Laplacian"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

# ----------------------------
# 3. Extract frames (sampling + filter blurry)
# ----------------------------
video_name = os.path.splitext(os.path.basename(args.video))[0]
frame_dir = f"data/frames/{video_name}"
os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(args.video)
frame_paths = []
frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % args.frame_step == 0:
        if not is_blurry(frame):  # skip blurry frames
            frame_path = os.path.join(frame_dir, f"frame_{saved_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_id += 1
    frame_id += 1

cap.release()
print(f"[INFO] Extracted {len(frame_paths)} frames (after filtering) to {frame_dir}")

# ----------------------------
# 4. Encode frames
# ----------------------------
images = [preprocess(Image.open(fp)).unsqueeze(0).to(device) for fp in frame_paths]
image_features = []

with torch.no_grad():
    for img in images:
        feat = model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
        image_features.append(feat.cpu().numpy())

image_features = np.vstack(image_features)
np.save(f"data/features/{video_name}.npy", image_features)
print(f"[INFO] Saved features to data/features/{video_name}.npy")

# ----------------------------
# 5. Encode query
# ----------------------------
text = clip.tokenize([args.query]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()

# ----------------------------
# 6. Compute similarities
# ----------------------------
similarities = (image_features @ text_features.T).squeeze()

# ----------------------------
# 7. Pick top-k frames (remove near-duplicates)
# ----------------------------
sorted_idx = np.argsort(similarities)[::-1]  # all frames sorted
selected = []
selected_idx = []

for idx in sorted_idx:
    feat = image_features[idx]
    # check similarity with already selected
    if all(cosine_similarity(feat, image_features[i]) < 0.95 for i in selected_idx):
        selected_idx.append(idx)
        selected.append(frame_paths[idx])
    if len(selected) >= args.topk:
        break

# ----------------------------
# 8. Save results (clear old outputs first)
# ----------------------------
output_dir = f"outputs/{video_name}"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

for i, frame_path in enumerate(selected):
    shutil.copy(frame_path, os.path.join(output_dir, f"keyframe_{i+1}.jpg"))

print(f"[INFO] Saved {len(selected)} keyframes to {output_dir}")
