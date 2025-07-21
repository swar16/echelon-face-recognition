# faiss_utils.py

import os
import faiss
import numpy as np

# dimensionality of your InsightFace embeddings
EMBEDDING_DIM = 512

# ========================================================
# write the index file in the same folder as this module
# ========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "face_index.faiss")


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build and return a FlatL2 FAISS index from an (N,512) array.
    """
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    return index


def save_faiss_index(index: faiss.IndexFlatL2) -> None:
    """
    Persist the given FAISS index to disk. 
    Ensures that BASE_DIR exists before writing.
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)


def load_faiss_index() -> faiss.IndexFlatL2 | None:
    """
    Load and return the FAISS index if it exists, else None.
    """
    if not os.path.exists(INDEX_PATH):
        return None
    return faiss.read_index(INDEX_PATH)
