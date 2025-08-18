from flask import Flask, request, send_file, jsonify
import tempfile
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import os

app = Flask(__name__)

# =========================
# 🔹 Global Variables for Lazy Loading
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_model = None
scene_model = None
face2paint = None
mtcnn = None

def load_models():
    """Load models only when needed"""
    global face_model, scene_model, face2paint, mtcnn
    
    if face_model is None:
        try:
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
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

# =========================
# 🔹 API Routes
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AnimeGANv2 Cartoonizer API",
        "status": "running",
        "endpoints": {
            "/anime-gan?style=face": "Anime style (best for portraits, sharp eyes/mouth)",
            "/anime-gan?style=paprika": "Anime style (best for full scenes, Studio Ghibli vibe)"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Service is running"})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "API is working!", "message": "AnimeGANv2 is ready"})

@app.route("/anime-gan", methods=["POST"])
def anime_gan():
    # Load models on first request
    try:
        load_models()
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500
    
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
