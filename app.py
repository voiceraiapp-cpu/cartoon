from flask import Flask, request, send_file, jsonify
import tempfile
import os
import requests
from gradio_client import Client, handle_file

app = Flask(__name__)

client = Client("VOICER12345/animegan-cartoonizer")

@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Call the Gradio API
        result = client.predict(
            img=handle_file(tmp_path),
            api_name="/predict"
        )

        # Gradio returns a string (path or URL), not a dict
        if result.startswith("http"):  # if it's a URL, download it
            response = requests.get(result)
            img_io = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_io.write(response.content)
            img_io.seek(0)
            return send_file(img_io.name, mimetype='image/png')
        else:
            # if it's a local path returned by Gradio
            return send_file(result, mimetype='image/png')
    finally:
        # Clean up the uploaded temp file
        os.remove(tmp_path)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AnimeGAN Cartoonizer API", "endpoint": "/cartoonize"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
