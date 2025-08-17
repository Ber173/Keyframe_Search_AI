import cv2
import os
from skimage.metrics import structural_similarity as ssim

def extract_frames(video_path, output_dir, threshold=0.9):
    """
    Extract unique frames from a video using SSIM-based similarity check.
    
    Args:
        video_path (str): Path to the input video
        output_dir (str): Directory to save frames
        threshold (float): SSIM threshold (0~1). 
                           Lower = keep more frames, Higher = keep fewer (default=0.9).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_gray = None
    frame_count, saved_count = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # Save first frame
            timestamp = frame_count / fps
            frame_name = f"frame_{saved_count:04d}_t{timestamp:.2f}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            prev_gray = gray
            saved_count += 1
        else:
            score, _ = ssim(prev_gray, gray, full=True)
            if score < threshold:  # Only save if different enough
                timestamp = frame_count / fps
                frame_name = f"frame_{saved_count:04d}_t{timestamp:.2f}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                prev_gray = gray
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[INFO] Extracted {saved_count} unique frames from {video_path} → {output_dir}")
    return saved_count
