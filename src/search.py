import faiss, torch, clip, numpy as np, os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def search_keyframe(query, features_path, img_dir, topk=3):
    # load features
    features = np.load(features_path).astype("float32")
    files = sorted(os.listdir(img_dir))

    # encode query
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text).cpu().numpy().astype("float32")

    # search
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features)
    D, I = index.search(text_feat, topk)

    return [os.path.join(img_dir, files[i]) for i in I[0]]
