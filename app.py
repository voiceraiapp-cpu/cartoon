from flask import Flask, request, send_file, jsonify
import tempfile
import torch
from PIL import Image
import numpy as np
import os
import sys

app = Flask(__name__)

# =========================
# 🔹 Global Variables for Lazy Loading
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paprika_model = None
face2paint = None

def download_models():
    """Download paprika model during build/startup"""
    print("🔄 Downloading paprika model...")
    try:
        # Set torch hub cache directory
        torch.hub.set_dir('./models')
        
        # Download model with retry logic
        import time
        max_retries = 3
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                print(f"📥 Attempt {attempt + 1}: Downloading paprika model...")
                paprika_model = torch.hub.load(
                    "bryandlee/animegan2-pytorch:main",
                    "generator",
                    pretrained="paprika",
                    force_reload=False  # Use cached if available
                )
                
                print(f"📥 Attempt {attempt + 1}: Downloading face2paint...")
                face2paint = torch.hub.load(
                    "bryandlee/animegan2-pytorch:main",
                    "face2paint",
                    size=512,
                    force_reload=False
                )
                
                print("✅ Paprika model downloaded successfully!")
                return paprika_model, face2paint
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e
                    
    except Exception as e:
        print(f"❌ Failed to download model after {max_retries} attempts: {e}")
        raise e

def load_models():
    """Load paprika model only when needed"""
    global paprika_model, face2paint
    
    if paprika_model is None:
        try:
            # Try to load from cache first
            torch.hub.set_dir('./models')
            
            # Check if models exist in cache
            cache_dir = torch.hub.get_dir()
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            print("🔄 Loading paprika model...")
            
            # Load paprika model (will use cache if available)
            paprika_model = torch.hub.load(
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
            
            print("✅ Paprika model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading paprika model: {e}")
            raise e

# =========================
# 🔹 Download models at startup (for production)
# =========================
if not os.environ.get('FLASK_ENV') == 'development':
    try:
        download_models()
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-download paprika model: {e}")
        print("🔄 Model will be downloaded on first request instead")

# =========================
# 🔹 API Routes
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AnimeGANv2 Paprika Cartoonizer API",
        "status": "running",
        "device": str(device),
        "model_loaded": paprika_model is not None,
        "style": "paprika (Studio Ghibli-like anime style)",
        "endpoints": {
            "/anime-gan": "Convert image to paprika anime style",
            "/cartoonize": "Alias for /anime-gan"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Paprika cartoonizer service is running",
        "model_loaded": paprika_model is not None,
        "device": str(device)
    })

@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "API is working!", 
        "message": "Paprika AnimeGANv2 is ready",
        "model_ready": paprika_model is not None
    })

@app.route("/load-models", methods=["POST"])
def load_models_endpoint():
    """Endpoint to manually trigger model loading"""
    try:
        load_models()
        return jsonify({"message": "Paprika model loaded successfully!", "status": "success"})
    except Exception as e:
        return jsonify({"error": f"Failed to load paprika model: {str(e)}"}), 500

@app.route("/anime-gan", methods=["POST"])
@app.route("/cartoonize", methods=["POST"])  # Alias for easier access
def anime_gan():
    # Load model on first request with better error handling
    try:
        if paprika_model is None:
            print("🔄 Loading paprika model on first request...")
            load_models()
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({
                "error": "Service temporarily unavailable due to model download limits. Please try again in a few minutes.",
                "details": "The paprika model is being downloaded and there's a temporary rate limit. This usually resolves quickly."
            }), 503
        else:
            return jsonify({"error": f"Failed to load paprika model: {error_msg}"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded. Please upload an image file."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format. Please upload a valid image file (JPG, PNG, etc.)"}), 400

    try:
        # Full-scene paprika stylization
        with torch.no_grad():
            final_img = face2paint(paprika_model, img)

        # Save to temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        final_img.save(temp.name)
        return send_file(temp.name, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": f"Cartoonization failed: {str(e)}"}), 500

# =========================
# 🔹 Main Runner
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
