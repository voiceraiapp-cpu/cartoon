from flask import Flask, request, send_file, jsonify
import tempfile
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import os
import sys

app = Flask(__name__)

# =========================
# 🔹 Global Variables for Lazy Loading
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_model = None
scene_model = None
face2paint = None
mtcnn = None

def download_models():
    """Download models during build/startup"""
    print("🔄 Downloading models...")
    try:
        # Set torch hub cache directory
        torch.hub.set_dir('./models')
        
        # Download models with retry logic
        import time
        max_retries = 3
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                print(f"📥 Attempt {attempt + 1}: Downloading face model...")
                face_model = torch.hub.load(
                    "bryandlee/animegan2-pytorch:main",
                    "generator",
                    pretrained="face_paint_512_v2",
                    force_reload=False  # Use cached if available
                )
                
                print(f"📥 Attempt {attempt + 1}: Downloading scene model...")
                scene_model = torch.hub.load(
                    "bryandlee/animegan2-pytorch:main",
                    "generator",
                    pretrained="paprika",
                    force_reload=False
                )
                
                print(f"📥 Attempt {attempt + 1}: Downloading face2paint...")
                face2paint = torch.hub.load(
                    "bryandlee/animegan2-pytorch:main",
                    "face2paint",
                    size=512,
                    force_reload=False
                )
                
                print("✅ All models downloaded successfully!")
                return face_model, scene_model, face2paint
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e
                    
    except Exception as e:
        print(f"❌ Failed to download models after {max_retries} attempts: {e}")
        raise e

def load_models():
    """Load models only when needed"""
    global face_model, scene_model, face2paint, mtcnn
    
    if face_model is None:
        try:
            # Try to load from cache first
            torch.hub.set_dir('./models')
            
            # Check if models exist in cache
            cache_dir = torch.hub.get_dir()
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            print("🔄 Loading models...")
            
            # Load models (will use cache if available)
            face_model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "generator",
                pretrained="face_paint_512_v2",
                force_reload=False
            ).to(device).eval()
            
            scene_model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "generator",
                pretrained="paprika",
                force_reload=False
            ).to(device).eval()
            
            face2paint = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "face2paint",
                size=512,
                force_reload=False
            )
            
            # Face detector
            mtcnn = MTCNN(keep_all=True, device=device)
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

# =========================
# 🔹 Download models at startup (for production)
# =========================
if not os.environ.get('FLASK_ENV') == 'development':
    try:
        download_models()
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-download models: {e}")
        print("🔄 Models will be downloaded on first request instead")

# =========================
# 🔹 API Routes
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AnimeGANv2 Cartoonizer API",
        "status": "running",
        "device": str(device),
        "models_loaded": face_model is not None,
        "endpoints": {
            "/anime-gan?style=face": "Anime style (best for portraits, sharp eyes/mouth)",
            "/anime-gan?style=paprika": "Anime style (best for full scenes, Studio Ghibli vibe)"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Service is running",
        "models_loaded": face_model is not None,
        "device": str(device)
    })

@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "API is working!", 
        "message": "AnimeGANv2 is ready",
        "models_ready": face_model is not None
    })

@app.route("/load-models", methods=["POST"])
def load_models_endpoint():
    """Endpoint to manually trigger model loading"""
    try:
        load_models()
        return jsonify({"message": "Models loaded successfully!", "status": "success"})
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

@app.route("/anime-gan", methods=["POST"])
def anime_gan():
    # Load models on first request with better error handling
    try:
        if face_model is None:
            print("🔄 Loading models on first request...")
            load_models()
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({
                "error": "Service temporarily unavailable due to model download limits. Please try again in a few minutes.",
                "details": "The AI models are being downloaded and there's a temporary rate limit. This usually resolves quickly."
            }), 503
        else:
            return jsonify({"error": f"Failed to load models: {error_msg}"}), 500
    
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
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
