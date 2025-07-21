### /mnt/data/recognize_webcam.py

# app/recognize_webcam.py
import cv2
import numpy as np
import insightface
from faiss_utils import load_faiss_index
from face_utils import load_embeddings

MATCH_THRESHOLD = 0.25  # slightly relaxed threshold for side profiles

# Use high-resolution multi-view capable face detection
MODEL = insightface.app.FaceAnalysis(name="buffalo_l")
MODEL.prepare(ctx_id=-1, det_size=(640, 480))

def recognize_face(embedding, names):
    index = load_faiss_index()
    if index is None or embedding is None:
        return "Unknown"
    D, I = index.search(embedding.reshape(1, -1).astype(np.float32), 1)
    dist, idx = D[0][0], I[0][0]
    print(dist)
    return names[idx] if dist < 1 - MATCH_THRESHOLD else "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    names, _ = load_embeddings()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = MODEL.get(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            emb = face.embedding
            emb /= np.linalg.norm(emb)
            emb = emb.astype(np.float32)

            name = recognize_face(emb, names)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



