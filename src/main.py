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


# create a fastapi app instance.
app = FastAPI(
    title="Lung Segmentation with TFLite",
    description="""A lung-segmentation app with Residual U-Net model saved as tflite model. 
    input is a CXR image, then you get a segmented mask as output.""",
    version="0.1.0",
    Author="pejmanS21")

# load the keras model.
model_runet = TFLiteModel("../weights/cxr_resunet.tflite")
model_unet = TFLiteModel("../weights/cxr_unet.tflite")


@app.post("/resunet-lite-JP")
async def resunet_lite_predict_jp(image: UploadFile = File(...)):
    uploaded_file = await image.read()
    segmentor = Segmentor(model=model_runet, uploaded_file=uploaded_file, dim=256)
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


@app.post("/resunet-lite")
async def resunet_lite_predict_visual(image: UploadFile = File(...)):
    """predict lung mask for CXR images

    Args:
        image (UploadFile, optional): [description]. Defaults to File(...).

    Predict: 
        A mask for uploadfile.
    Returns:
        predicted mask as Bytes. (No need to save, then show images)
    """
    uploaded_file = await image.read()
    segmentor = Segmentor(model=model_runet, uploaded_file=uploaded_file, dim=256)
    image = segmentor.preprocess(pre_process=True)
    mask = segmentor.predictions(image)
    
    _, mask_png = cv2.imencode(".png", mask[0] * 255.0)
    return StreamingResponse(io.BytesIO(mask_png.tobytes()), media_type="image/png")


@app.post("/unet_lite")
async def unet_lite_predict_visual(image: UploadFile = File(...)):
    """predict lung mask for CXR images

    Args:
        image (UploadFile, optional): [description]. Defaults to File(...).

    Predict: 
        A mask for uploadfile.
    Returns:
        predicted mask as Bytes. (No need to save, then show images)
    """

    uploaded_file = await image.read()
    segmentor = Segmentor(model=model_unet, uploaded_file=uploaded_file, dim=256)
    image = segmentor.preprocess(pre_process=True)
    mask = segmentor.predictions(image)
    
    _, mask_png = cv2.imencode(".png", mask[0] * 255.0)
    return StreamingResponse(io.BytesIO(mask_png.tobytes()), media_type="image/png")


