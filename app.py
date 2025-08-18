from flask import Flask, request, send_file, jsonify
import tempfile
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN

app = Flask(__name__)

# =========================
# 🔹 Load AnimeGANv2 Models
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face-focused model (sharp eyes/mouth)
face_model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2"
).to(device).eval()

# Full-scene model (Paprika style)
scene_model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="paprika"
).to(device).eval()

# Helper
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "face2paint",
    size=512
)

# Face detector
mtcnn = MTCNN(keep_all=True, device=device)


# =========================
# 🔹 API Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AnimeGANv2 Cartoonizer API",
        "endpoints": {
            "/anime-gan?style=face": "Anime style (best for portraits, sharp eyes/mouth)",
            "/anime-gan?style=paprika": "Anime style (best for full scenes, Studio Ghibli vibe)"
        }
    })


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "API is working!", "message": "AnimeGANv2 is ready"})


@app.route("/anime-gan", methods=["POST"])
def anime_gan():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    # Choose style
    style = request.args.get("style", "face")
    model = face_model if style == "face" else scene_model

    try:
        img = Image.open(file.stream).convert("RGB")
        np_img = np.array(img)
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    try:
        if style == "face":
            # Detect faces
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]

                    # Crop face
                    face_crop = img.crop((x1, y1, x2, y2))

                    # Stylize face
                    with torch.no_grad():
                        stylized_face = face2paint(model, face_crop)

                    # Resize to original face size
                    stylized_face = stylized_face.resize((x2-x1, y2-y1))

                    # Paste back
                    np_face = np.array(stylized_face)
                    np_img[y1:y2, x1:x2] = np_face

            final_img = Image.fromarray(np_img)

        else:
            # Full-scene stylization
            with torch.no_grad():
                final_img = face2paint(model, img)

        # Save to temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        final_img.save(temp.name)
        return send_file(temp.name, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# =========================
# 🔹 Main Runner
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
