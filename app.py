from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tempfile

app = Flask(__name__)

def cartoonize_image(img):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Edge mask
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )

    # Smooth color
    color = cv2.bilateralFilter(img, 9, 250, 250)

    # Combine edges + color
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cartoonizer API is running! Use POST /cartoonize with an image."})

@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    cartoon = cartoonize_image(img)

    # Save to temp file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp.name, cartoon)

    return send_file(temp.name, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
