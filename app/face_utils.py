# app/face_utils.py

import insightface
import numpy as np
import sqlite3
import os
from app.faiss_utils import create_faiss_index, save_faiss_index

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Initialize InsightFace
MODEL = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
MODEL.prepare(ctx_id=0)

def init_db():
    """Create users and embeddings tables if they don’t exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT    UNIQUE NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            embedding BLOB    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def extract_embedding(image_bgr):
    """Run detection+embedding; returns 512-d normalized embedding or None."""
    faces = MODEL.get(image_bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)
    return emb.astype(np.float32)

def _validate_pose(image_bgr, expected_pose):
    """
    Validate that the main face matches expected_pose:
      - 'front': nose near bbox center
      - 'right': nose shifted right of center
      - 'left' : nose shifted left of center
    """
    faces = MODEL.get(image_bgr)
    if not faces:
        return False
    faces.sort(key=lambda f: f.det_score, reverse=True)
    face = faces[0]
    x1, _, x2, _ = face.bbox.astype(float)
    mid_x = (x1 + x2) / 2
    nose_x = face.kps[2][0]  # landmark #2 is nose tip
    width = x2 - x1
    offset = width * 0.1

    if expected_pose == 'front':
        return abs(nose_x - mid_x) < offset
    elif expected_pose == 'right':
        return (nose_x - mid_x) > offset
    elif expected_pose == 'left':
        return (mid_x - nose_x) > offset
    return False

def validate_and_extract(image_bgr, pose):
    """
    Returns embedding if pose matches and face detected, else None.
    pose ∈ {'front','right','left'}
    """
    if not _validate_pose(image_bgr, pose):
        return None
    return extract_embedding(image_bgr)

def save_embedding(name, embedding_list):
    """Store embeddings in SQLite and rebuild FAISS index."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
    c.execute("SELECT id FROM users WHERE name = ?", (name,))
    user_id = c.fetchone()[0]

    for emb in embedding_list:
        c.execute(
            "INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
            (user_id, emb.tobytes())
        )
    conn.commit()
    conn.close()
    rebuild_faiss_index()

def load_embeddings():
    """Returns (names_list, np.ndarray of embeddings)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
      SELECT u.name, e.embedding
        FROM embeddings e
        JOIN users u ON e.user_id = u.id
    """)
    rows = c.fetchall()
    conn.close()

    names, embs = [], []
    for name, blob in rows:
        names.append(name)
        embs.append(np.frombuffer(blob, dtype=np.float32))
    return names, np.stack(embs, axis=0) if embs else (names, np.zeros((0, 512), dtype=np.float32))

def rebuild_faiss_index():
    """Rebuild and persist the FAISS index from all stored embeddings."""
    names, embs = load_embeddings()
    if embs.shape[0] == 0:
        return
    index = create_faiss_index(embs)
    save_faiss_index(index)
    print("[FAISS] Index rebuilt.")
