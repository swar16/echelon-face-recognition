# streamlit_app.py

import os
import tempfile

import streamlit as st
from PIL import Image
import numpy as np

from face_utils import (
    extract_embedding,
    validate_and_extract,
    save_embedding,
    load_embeddings,
)
from faiss_utils import load_faiss_index

# match threshold: 1 - distance (tweak to your accuracy/speed needs)
MATCH_THRESHOLD = 0.5

st.set_page_config(
    page_title="Echelon Face Recognition",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Echelon Face Recognition Demo")

mode = st.radio("Choose mode", ["Enroll", "Recognize"], index=1)

uploaded_file = st.file_uploader(
    "Upload a face image (.jpg / .jpeg / .png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Save upload to a temp file so PIL can open it
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tf.write(uploaded_file.read())
    tf.close()

    # Load as RGB numpy array
    pil_img = Image.open(tf.name).convert("RGB")
    img = np.array(pil_img)

    if mode == "Enroll":
        name = st.text_input("Name for enrollment")
        if st.button("Enroll"):
            # Validate front‐pose & extract embedding
            emb = validate_and_extract(img, pose="front")
            if emb is None:
                st.error("No front‐facing face detected. Please try again.")
            else:
                save_embedding(name, [emb])
                st.success(f"Enrolled “{name}” successfully!")
    else:  # Recognize
        if st.button("Identify"):
            emb = extract_embedding(img)
            names, embeddings = load_embeddings()
            index = load_faiss_index()

            if index is None or not names:
                st.warning("No faces enrolled yet. Switch to Enroll mode first.")
            else:
                # Search for the closest match
                D, I = index.search(emb.reshape(1, -1).astype(np.float32), 1)
                dist, idx = D[0][0], I[0][0]
                score = 1.0 - dist

                if score >= MATCH_THRESHOLD:
                    st.success(f"👋 Hello, **{names[idx]}**!  (score {score:.2f})")
                else:
                    st.error("No confident match found.")
    # clean up temp file
    os.unlink(tf.name)
else:
    st.info("Please upload a face image to get started.")
