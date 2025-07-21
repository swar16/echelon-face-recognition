# streamlit_app.py

import os
import tempfile

import streamlit as st
import cv2
import numpy as np

from face_utils import extract_embedding, load_embeddings, save_embedding, validate_and_extract
from faiss_utils import load_faiss_index

# thresholds & paths
MATCH_THRESHOLD = 0.5  # tweak if you like

st.title("Echelon Face-Recognition Demo")

mode = st.radio("Mode", ["Enroll", "Recognize"])

uploaded = st.file_uploader("Upload a face image (.jpg/.png)", type=["jpg", "jpeg", "png"])
if uploaded:
    # Save to a temp file so OpenCV can read it
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tf.write(uploaded.read())
    tf.close()
    img = cv2.imread(tf.name)

    if img is None:
        st.error("Could not read image.")
    else:
        if mode == "Enroll":
            name = st.text_input("Name for enrollment")
            if name and st.button("Enroll"):
                # Here we assume front-facing pose; adjust if you want right/left
                emb = validate_and_extract(img, pose="front")
                if emb is None:
                    st.error("No face or wrong pose detected. Try a clear, front-facing photo.")
                else:
                    save_embedding(name, [emb])
                    st.success(f"Enrolled “{name}” ✔️")
        else:  # Recognize
            if st.button("Identify"):
                emb = extract_embedding(img)
                names, _ = load_embeddings()
                index = load_faiss_index()
                if index is None or len(names)==0:
                    st.warning("No enrolled faces yet.")
                else:
                    D, I = index.search(emb.reshape(1,-1).astype(np.float32), 1)
                    dist, idx = D[0][0], I[0][0]
                    score = 1 - dist
                    if score >= MATCH_THRESHOLD:
                        st.success(f"Hello, **{names[idx]}**!  (score {score:.2f})")
                    else:
                        st.error("No good match found.")
    os.unlink(tf.name)
