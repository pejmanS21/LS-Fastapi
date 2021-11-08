from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tflite_loader import TFLiteModel
from segmentation import Segmentor
import cv2
import os
import numpy as np
from PIL import Image
from io import BytesIO


# create a fastapi app instance.
app = FastAPI(
    title="Lung Segmentation with TFLite",
    description="""A lung-segmentation app with Residual U-Net model saved as tflite model. 
    input is a CXR image, then you get a segmented mask as output.""",
    version="0.1.0",
    Author="pejmanS21")

# load the keras model.
model = TFLiteModel("../weights/cxr_resunet.tflite")


@app.post("/resunet-lite")
async def resunet_lite_predict(image: UploadFile = File(...)):
    uploaded_file = await image.read()
    segmentor = Segmentor(model=model, uploaded_file=uploaded_file, dim=256)
    image = segmentor.preprocess(pre_process=True)
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