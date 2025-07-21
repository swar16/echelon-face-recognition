import os
import shutil
import tempfile

import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename    # ← NEW import

from face_utils import init_db, save_embedding, validate_and_extract

app = Flask(__name__)
BASE_DIR       = os.path.dirname(__file__)
ENROLLED_DIR   = os.path.join(BASE_DIR, "enrolled_images")
TEMP_PARENT    = os.path.join(BASE_DIR, "temp")

# ensure directories exist
os.makedirs(ENROLLED_DIR, exist_ok=True)
os.makedirs(TEMP_PARENT, exist_ok=True)

# Initialize DB on startup
init_db()

@app.route("/", methods=["GET", "POST"])
def upload():
    message = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            message = "Name is required."
            return render_template("upload.html", message=message)

        # gather files
        slots = {
            'front': request.files.get("front_image"),
            'right': request.files.get("right_image"),
            'left':  request.files.get("left_image"),
        }

        temp_dir = tempfile.mkdtemp(dir=TEMP_PARENT)
        embeddings = []
        invalid = []

        try:
            for pose, file in slots.items():
                if not file or file.filename == "":
                    invalid.append(f"{pose} image missing")
                    continue

                # SECURE and prefix the filename
                clean_name = secure_filename(file.filename)
                filename = f"{pose}_{clean_name}"
                path = os.path.join(temp_dir, filename)
                file.save(path)

                img = cv2.imread(path)
                if img is None:
                    invalid.append(f"{pose}: unreadable image")
                    continue

                emb = validate_and_extract(img, pose)
                if emb is None:
                    invalid.append(f"{pose}: invalid pose or face not detected")
                else:
                    embeddings.append(emb)

            if len(embeddings) == 3:
                user_dir = os.path.join(ENROLLED_DIR, name)
                os.makedirs(user_dir, exist_ok=True)
                for fname in os.listdir(temp_dir):
                    shutil.move(
                        os.path.join(temp_dir, fname),
                        os.path.join(user_dir, fname)
                    )
                save_embedding(name, embeddings)
                message = f"✅ {name} enrolled with 3 valid images."
            else:
                message = "❌ Enrollment failed:\n" + "\n".join(invalid)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return render_template("upload.html", message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
