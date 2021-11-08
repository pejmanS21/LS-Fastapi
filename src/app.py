from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tflite_loader import TFLiteModel
from segmentation import Segmentor
import cv2
import os
import numpy as np
from PIL import Image
import io
from starlette.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel


app = FastAPI(
    title="Lung Segmentation with TFLite",
    description="""A lung-segmentation app with Residual U-Net model saved as tflite model. 
    input is a CXR image, then you get a segmented mask as output.""",
    version="0.1.0",
    Author="pejmanS21")

# load the keras model.
model_runet = TFLiteModel("../weights/cxr_resunet.tflite")
model_unet = TFLiteModel("../weights/cxr_unet.tflite")


@app.post("/resunet/{output_type}")
async def resunet(output_type: str, pre_process: Optional[str] = None, image: UploadFile = File(...)):
    """[summary]

    Args:<br>
        output_type (str): 1 ("Bytes") for visualize here, or 2 ("Saved") to visualize in `/imshow` endpoint.<br>
        pre_process (Optional[str], optional): [[y] or [Y] to use `DHE`. Defaults to None.<br> 
        image (UploadFile, optional): [description]. Defaults to File(...).<br>

    Returns:<br>
        if output_type == "Bytes" or "1":<br>
            return `image`.<br><br>
        elif output_type == "Saved" or "2":<br>
            return `dict`.<br>
    """

    if pre_process:
        pre_process = True
    
    if output_type == "Bytes" or output_type == "1":
        uploaded_file = await image.read()
        segmentor = Segmentor(model=model_runet, uploaded_file=uploaded_file, dim=256)
        image = segmentor.preprocess(pre_process=pre_process)
        mask = segmentor.predictions(image)
        
        _, mask_png = cv2.imencode(".png", mask[0] * 255.0)
        return StreamingResponse(io.BytesIO(mask_png.tobytes()), media_type="image/png")

    elif output_type == "Saved" or output_type == "2":
        uploaded_file = await image.read()
        segmentor = Segmentor(model=model_runet, uploaded_file=uploaded_file, dim=256)
        image = segmentor.preprocess(pre_process=pre_process)
        mask = segmentor.predictions(image)

        return {"message": "Masked Saved Successfully!",
                "mask_url": "http://127.0.0.1:8000/imshow", 
                "mask_shape": str(mask.shape)}


@app.post("/unet/{output_type}")
async def unet(output_type: str, pre_process: Optional[str] = None, image: UploadFile = File(...)):
    """[summary]

    Args:<br>
        output_type (str): 1 ("Bytes") for visualize here, or 2 ("Saved") to visualize in `/imshow` endpoint.<br>
        pre_process (Optional[str], optional): [[y] or [Y] to use `DHE`. Defaults to None.<br> 
        image (UploadFile, optional): [description]. Defaults to File(...).<br>

    Returns:<br>
        if output_type == "Bytes" or "1":<br>
            return `image`.<br><br>
        elif output_type == "Saved" or "2":<br>
            return `dict`.<br>
    """

    if pre_process:
        pre_process = True
    
    if output_type == "Bytes" or output_type == "1":
        uploaded_file = await image.read()
        segmentor = Segmentor(model=model_unet, uploaded_file=uploaded_file, dim=256)
        image = segmentor.preprocess(pre_process=pre_process)
        mask = segmentor.predictions(image)
        
        _, mask_png = cv2.imencode(".png", mask[0] * 255.0)
        return StreamingResponse(io.BytesIO(mask_png.tobytes()), media_type="image/png")

    elif output_type == "Saved" or output_type == "2":
        uploaded_file = await image.read()
        segmentor = Segmentor(model=model_unet, uploaded_file=uploaded_file, dim=256)
        image = segmentor.preprocess(pre_process=pre_process)
        mask = segmentor.predictions(image)

        return {"message": "Masked Saved Successfully!",
                "mask_url": "http://127.0.0.1:8000/imshow", 
                "mask_shape": str(mask.shape)}


@app.get("/imshow")
async def imshow():
    mask_path = "../images/mask.png"
    if os.path.isfile(mask_path):
        return FileResponse(mask_path)

@app.get("/")
async def main():
    return {"message": "http://127.0.0.1:8000/docs"}
