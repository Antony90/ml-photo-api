from base64 import b64decode
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from binascii import Error as DecodeError

def base64_img_to_array(img_data):
    try:
        img_base64 = img_data.split(",")[1]
    except AttributeError:
        raise HTTPException(detail=f"Entry in 'images' list is not a base64 image: '{img_data[:5]}'", status_code=400)
    except IndexError:
        img_base64 = img_data
    
    try:
        decoded_img = b64decode(img_base64)
    except DecodeError:
        raise HTTPException(detail=f"Entry in 'images' list is not a base64 string", status_code=500)
    try:
        img = Image.open(BytesIO(decoded_img))
    except UnidentifiedImageError:
        raise HTTPException(detail="Unsupported file extension. Use .jpg or .png", status_code=400)
        
    if img.mode == 'RGBA':
        img = img.convert("RGB")
    elif img.mode == 'CMYK':
        img = img.convert('CMYK')
    # elif not img.mode == 'RGB':
        raise HTTPException(detail="Unsupported color mode {img.mode}. Accepted: RGB, RGBA and CMYK", status_code=400)
    
    return np.array(img)

def images_to_arrays(base64_images: list[str]):
    img_batch: list[np.ndarray] = []
    for img_data in base64_images:
        img_arr = base64_img_to_array(img_data)

        if len(img_arr.shape) == 2:
            raise HTTPException(detail=f"Image has a single color channel. Expected RGB.", status_code=400)
        else:
            img_batch.append(img_arr)
    return img_batch