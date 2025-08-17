import cv2, os

def extract_keyframes(video_path, out_dir, interval=30):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx, saved_idx = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(out_dir, f"{saved_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    print(f"[INFO] Extracted {saved_idx} frames to {out_dir}")
