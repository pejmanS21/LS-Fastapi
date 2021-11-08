from PIL import Image, ImageOps
import os
import io
import cv2
import numpy as np


class Segmentor:
    def __init__(self, model, uploaded_file, dim: int = 256):
        self.model = model
        self.uploaded_file = uploaded_file
        self.dim = dim
    
    def preprocess(self, pre_process:bool = False):
        reference_shape = (self.dim, self.dim, 1)

        image = Image.open(io.BytesIO(self.uploaded_file)).convert("L")
        image = ImageOps.fit(image, (self.dim, self.dim))
        image = np.asarray(image)
        
        # pil to cv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # check channel
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # compatible size
            if image.shape != reference_shape:
                image = cv2.resize(image, (self.dim, self.dim))

        if pre_process:
            image = cv2.equalizeHist(image)

        image = image.reshape(1, self.dim, self.dim, 1)
        image = (image - 127.0) / 127.0
        
        return image

    def predictions(self, image):
        mask = self.model.predict(image)
        cv2.imwrite("../images/mask.png", mask[0] * 255.0)
        return mask