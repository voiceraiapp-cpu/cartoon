from cartoon_diffusion import CartoonDiffusionPipeline
import gradio as gr
from fastapi import FastAPI
from PIL import Image
import io

app = FastAPI()

# Initialize the pipeline
pipeline = CartoonDiffusionPipeline.from_pretrained("wizcodes12/image_to_cartoonify")

# Gradio function for the interface
def generate_cartoon(image, hair_color=0.5, glasses=0.0, facial_hair=0.0):
    cartoon = pipeline(image, hair_color=hair_color, glasses=glasses, facial_hair=facial_hair)
    return cartoon

# FastAPI endpoint for API access
@app.post("/cartoonify")
async def cartoonify(image: bytes = File(...)):
    img = Image.open(io.BytesIO(image))
    cartoon = pipeline(img)
    output = io.BytesIO()
    cartoon.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")

# Gradio interface
interface = gr.Interface(
    fn=generate_cartoon,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(0, 1, value=0.5, label="Hair Color"),
        gr.Slider(0, 1, value=0.0, label="Glasses"),
        gr.Slider(0, 1, value=0.0, label="Facial Hair")
    ],
    outputs=gr.Image(type="pil"),
    title="Cartoon Generator"
)

if __name__ == "__main__":
    interface.launch()
