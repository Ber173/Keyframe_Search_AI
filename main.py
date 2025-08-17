import os
import cv2
import torch
import clip
import numpy as np
import shutil
import argparse
from PIL import Image

# ----------------------------
# 1. Parse arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, required=True, help="Path to directory containing videos")
parser.add_argument("--query", type=str, required=True, help="Text query to search in videos")
parser.add_argument("--topk", type=int, default=5, help="Number of keyframes to save per video")
args = parser.parse_args()

# ----------------------------
# 2. Load CLIP model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------
# 3. Encode query
# ----------------------------
text = clip.tokenize([args.query]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()

# ----------------------------
# 4. Process videos
# ----------------------------
video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith((".mp4", ".avi", ".mov"))]

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = f"data/frames/{video_name}"
    os.makedirs(frame_dir, exist_ok=True)

    feature_path = f"data/features/{video_name}.npy"
    frame_paths_file = f"data/features/{video_name}_frames.npy"

    # ----------------------------
    # 4a. Load cache if available
    # ----------------------------
    if os.path.exists(feature_path) and os.path.exists(frame_paths_file):
        print(f"[INFO] Found cached features for {video_name}, skipping extraction...")
        image_features = np.load(feature_path)
        frame_paths = np.load(frame_paths_file, allow_pickle=True).tolist()
    else:
        # ----------------------------
        # Extract frames
        # ----------------------------
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frame_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append((frame_path, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0))  # lưu thêm timestamp
            frame_id += 1

        cap.release()
        print(f"[INFO] Extracted {len(frame_paths)} frames from {video_name}")

        # ----------------------------
        # Encode frames
        # ----------------------------
        images = [preprocess(Image.open(fp[0])).unsqueeze(0).to(device) for fp in frame_paths]
        image_features = []

        with torch.no_grad():
            for img in images:
                feat = model.encode_image(img)
                feat /= feat.norm(dim=-1, keepdim=True)
                image_features.append(feat.cpu().numpy())

        image_features = np.vstack(image_features)

        # Save cache
        os.makedirs("data/features", exist_ok=True)
        np.save(feature_path, image_features)
        np.save(frame_paths_file, np.array(frame_paths, dtype=object))
        print(f"[INFO] Saved features for {video_name} to cache")

    # ----------------------------
    # 5. Compute similarities
    # ----------------------------
    similarities = (image_features @ text_features.T).squeeze()

    # ----------------------------
    # 6. Pick top-k frames
    # ----------------------------
    top_indices = np.argsort(similarities)[-args.topk:][::-1]
    top_k_frames = [frame_paths[i] for i in top_indices]

    # ----------------------------
    # 7. Save results
    # ----------------------------
    output_dir = f"outputs/{video_name}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, (frame_path, timestamp) in enumerate(top_k_frames):
        out_name = f"keyframe_{i+1}_at_{timestamp:.2f}s.jpg"
        shutil.copy(frame_path, os.path.join(output_dir, out_name))

    print(f"[INFO] Saved {len(top_k_frames)} keyframes to {output_dir}")
