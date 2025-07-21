# app/recognize_image.py
import sys
import cv2
import numpy as np
from face_utils import extract_embedding, load_embeddings
from faiss_utils import load_faiss_index

MATCH_THRESHOLD = 0.5

def recognize(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Invalid image")
        return
    emb = extract_embedding(img)
    names, _ = load_embeddings()
    index = load_faiss_index()

    if index is None or emb is None:
        print("Recognition failed.")
        return

    D, I = index.search(emb.reshape(1, -1).astype(np.float32), 1)
    dist, idx = D[0][0], I[0][0]

    if dist < 1 - MATCH_THRESHOLD:
        print(f"[MATCH] {names[idx]} (score={1 - dist:.2f})")
    else:
        print("No match")

if __name__ == "__main__":
    recognize(sys.argv[1])



