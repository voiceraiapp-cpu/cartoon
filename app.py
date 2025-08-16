from cartoon_diffusion import CartoonDiffusionPipeline
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from starlette.responses import Response

app = FastAPI()

# Initialize the pipeline
pipeline = CartoonDiffusionPipeline.from_pretrained("wizcodes12/image_to_cartoonify")

@app.get("/")
async def root():
    return {"message": "Cartoonify API is running!"}

@app.post("/cartoonify")
async def cartoonify(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        img = Image.open(file.file)
        cartoon = pipeline(img)
        output = io.BytesIO()
        cartoon.save(output, format="PNG")
        return Response(content=output.getvalue(), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

@app.post("/cartoonify-advanced")
async def cartoonify_advanced(
    file: UploadFile = File(...),
    hair_color: float = 0.5,
    glasses: float = 0.0,
    facial_hair: float = 0.0
):
    try:
        img = Image.open(file.file)
        cartoon = pipeline(
            img,
            hair_color=hair_color,
            glasses=glasses,
            facial_hair=facial_hair,
            num_inference_steps=50,
            guidance_scale=7.5
        )
        output = io.BytesIO()
        cartoon.save(output, format="PNG")
        return Response(content=output.getvalue(), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
