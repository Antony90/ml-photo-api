from base64 import b64decode
from PIL import Image, UnidentifiedImageError
from io import BytesIO

def base64_img_to_array(img_data):
    try:
        img_base64 = img_data.split(",")[1]
    except AttributeError:
        raise Exception(f"Entry in 'images' list is not a base64 image: '{img_data[:5]}'")
    
    decoded_img = b64decode(img_base64)
    try:
        img = Image.open(BytesIO(decoded_img))
    except UnidentifiedImageError:
        raise Exception("Unsupported file extension. Use .jpg or .png")
        
    if img.mode == 'RGBA':
        img = img.convert("RGB")
    elif img.mode == 'CMYK':
        img = img.convert('CMYK')
    # elif not img.mode == 'RGB':
        raise Exception("Unsupported color mode {img.mode}. Accepted: RGB, RGBA and CMYK")