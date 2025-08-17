import os, torch, clip, numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_images(img_dir, out_path):
    feats, files = [], sorted(os.listdir(img_dir))
    for f in files:
        img = preprocess(Image.open(os.path.join(img_dir, f))).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        feats.append(feat.cpu().numpy())
    feats = np.vstack(feats)
    np.save(out_path, feats)
    print(f"[INFO] Saved features to {out_path}")
    return feats, files
