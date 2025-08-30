import os
import cv2
import argparse
import numpy as np
import torch
import clip
from PIL import Image
from typing import List, Tuple
import shutil
import csv
import json

# -----------------------------
# 1) Model & device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# -----------------------------
# 2) Extract frames by target FPS (uniform sampling)
# -----------------------------
def extract_frames_uniform(video_path: str, frame_dir: str, fps_target: float = 1.0) -> List[float]:
    """
    Trích khung hình theo thời gian đều: ~fps_target frame/giây.
    Trả về danh sách timestamp (giây) tương ứng với mỗi frame đã lưu.
    """
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        # fallback đơn giản: đọc tất cả, ghi mỗi N khung
        src_fps = 25.0
    step = max(1, int(round(src_fps / max(1e-6, fps_target))))

    timestamps = []
    frame_idx_keep = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            ts = count / src_fps
            out_path = os.path.join(frame_dir, f"frame_{frame_idx_keep:04d}.jpg")
            cv2.imwrite(out_path, frame)
            timestamps.append(ts)
            frame_idx_keep += 1

        count += 1

    cap.release()
    return timestamps


# -----------------------------
# 3) Encode frames (batched CLIP)
# -----------------------------
def encode_frames_clip(frame_dir: str, batch_size: int = 32) -> np.ndarray:
    """
    Đọc tất cả .jpg trong frame_dir, encode bằng CLIP theo batch.
    Trả về mảng (N, D) đã L2-normalize.
    """
    files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")])
    if not files:
        return np.zeros((0, 512), dtype=np.float32)

    feats = []
    batch_imgs = []

    def flush_batch(images: List[torch.Tensor]):
        if not images:
            return None
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    for i, fname in enumerate(files, 1):
        img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
        batch_imgs.append(preprocess(img))
        if len(batch_imgs) == batch_size:
            feats.append(flush_batch(batch_imgs))
            batch_imgs = []

    if batch_imgs:
        feats.append(flush_batch(batch_imgs))

    return np.vstack(feats)


# -----------------------------
# 4) Load or build cache for a video
# -----------------------------
def prepare_video_features(video_path: str, fps_target: float) -> Tuple[str, np.ndarray, List[float]]:
    """
    Với 1 video:
    - Nếu đã có cache: load features + timestamps
    - Nếu chưa: trích frame đều theo fps_target -> encode -> lưu cache
    Trả về: (video_name, features_np, timestamps_list)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join("data", "frames", video_name)
    feat_dir = os.path.join("data", "features")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join("data", "frames"), exist_ok=True)

    feat_path = os.path.join(feat_dir, f"{video_name}.npy")
    ts_path = os.path.join(feat_dir, f"{video_name}_ts.npy")

    if os.path.exists(feat_path) and os.path.exists(ts_path):
        print(f"[INFO] Cache found for {video_name} -> load features")
        features = np.load(feat_path)
        timestamps = np.load(ts_path).tolist()
        return video_name, features, timestamps

    print(f"[INFO] Extract frames (uniform) for {video_name} @ {fps_target} fps")
    timestamps = extract_frames_uniform(video_path, frame_dir, fps_target=fps_target)

    print(f"[INFO] Encode frames for {video_name} (N={len(timestamps)})")
    features = encode_frames_clip(frame_dir)

    np.save(feat_path, features)
    np.save(ts_path, np.array(timestamps, dtype=np.float32))
    print(f"[INFO] Saved cache: {feat_path} & {ts_path}")

    return video_name, features, timestamps


# -----------------------------
# 5) Global search (top-k across ALL videos)
# -----------------------------
def global_search(video_pack: List[Tuple[str, np.ndarray, List[float]]],
                  query: str,
                  topk: int,
                  output_dir: str = "outputs") -> List[dict]:
    """
    Tính similarity giữa tất cả frame (từ nhiều video) với query.
    Lấy top-k toàn cục, copy ảnh ra outputs + trả metadata.
    """
    # reset outputs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # encode query
    with torch.no_grad():
        q = model.encode_text(clip.tokenize([query]).to(device))
        q = q / q.norm(dim=-1, keepdim=True)
    q = q.cpu().numpy()  # (1, D)

    # gom tất cả
    all_rows = []  # (video_name, frame_idx, ts, score)
    for video_name, feats, ts_list in video_pack:
        if feats.shape[0] == 0:
            continue
        sims = (feats @ q.T).squeeze()  # (N,)
        for idx, (score, ts) in enumerate(zip(sims, ts_list)):
            all_rows.append((video_name, idx, float(ts), float(score)))

    if not all_rows:
        print("[WARN] Không có frame nào để search.")
        return []

    all_rows.sort(key=lambda r: -r[3])  # sort theo score giảm dần
    all_rows = all_rows[:topk]

    # copy hình + build metadata
    results = []
    for rank, (vid, fidx, ts, score) in enumerate(all_rows, 1):
        src = os.path.join("data", "frames", vid, f"frame_{fidx:04d}.jpg")
        name = f"top{rank}_{vid}_t{ts:.2f}_s{score:.3f}.jpg"
        dst = os.path.join(output_dir, name)
        if os.path.exists(src):
            shutil.copy(src, dst)
        results.append({
            "rank": rank,
            "video": vid,
            "frame_index": fidx,
            "time_seconds": round(ts, 3),
            "score": round(score, 6),
            "output_file": dst
        })

    # lưu JSON & CSV
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "video", "frame_index", "time_seconds", "score", "output_file"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    return results


# -----------------------------
# 6) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Thư mục chứa video (.mp4/.avi/.mov)")
    parser.add_argument("--query", type=str, required=True, help="Câu truy vấn văn bản (tiếng Anh càng tốt)")
    parser.add_argument("--topk", type=int, default=5, help="Số ảnh tốt nhất toàn cục cần lấy")
    parser.add_argument("--fps", type=float, default=1.0, help="Target FPS khi trích khung hình (mặc định 1 fps)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size khi encode (chỉ dùng nội bộ)")
    args = parser.parse_args()

    # Liệt kê video
    exts = (".mp4", ".avi", ".mov", ".mkv")
    videos = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.lower().endswith(exts)]
    if not videos:
        raise RuntimeError(f"Không tìm thấy video trong {args.video_dir}")

    # Chuẩn bị cache features cho từng video
    video_pack = []
    for vp in videos:
        vid, feats, ts = prepare_video_features(vp, fps_target=args.fps)
        video_pack.append((vid, feats, ts))

    # Tìm top-k toàn cục
    results = global_search(video_pack, args.query, topk=args.topk, output_dir="outputs")

    print("\n=== TOP RESULTS ===")
    for r in results:
        print(f"[{r['rank']}] {r['video']} | frame {r['frame_index']} | t={r['time_seconds']:.2f}s | score={r['score']:.3f}")
    print(f"\nẢnh + metadata đã lưu ở: outputs/")


if __name__ == "__main__":
    # Mẹo cho macOS nếu gặp lỗi OpenMP: chạy với KMP_DUPLICATE_LIB_OK=TRUE ở lệnh bash
    main()
