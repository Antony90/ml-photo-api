from base64 import b64decode
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import InternalServerError
from binascii import Error as DecodeError

def base64_img_to_array(img_data):
    try:
        img_base64 = img_data.split(",")[1]
    except AttributeError:
        raise InternalServerError(f"Entry in 'images' list is not a base64 image: '{img_data[:5]}'")
    except IndexError:
        img_base64 = img_data
    
    try:
        decoded_img = b64decode(img_base64)
    except DecodeError:
        raise InternalServerError(f"Entry in 'images' list is not a base64 string")
    try:
        img = Image.open(BytesIO(decoded_img))
    except UnidentifiedImageError:
        raise InternalServerError("Unsupported file extension. Use .jpg or .png")
        
    if img.mode == 'RGBA':
        img = img.convert("RGB")
    elif img.mode == 'CMYK':
        img = img.convert('CMYK')
    # elif not img.mode == 'RGB':
        raise InternalServerError("Unsupported color mode {img.mode}. Accepted: RGB, RGBA and CMYK")
    
    return np.array(img)

def images_to_arrays(base64_images: list[str]):
    img_batch: list[np.ndarray] = []
    for img_data in base64_images:
        img_arr = base64_img_to_array(img_data)

        if len(img_arr.shape) == 2:
            raise InternalServerError(f"Image has a single color channel. Expected RGB.")
        else:
            img_batch.append(img_arr)
    return img_batch