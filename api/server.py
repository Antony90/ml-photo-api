from flask import Flask, request
from flask_cors import CORS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

from base64 import b64decode
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import json

# load model and category data
model = tf.keras.models.load_model('./models/loss__0.33__acc__0.74')
print(f"Loaded model")
with open('./dataset/categories.json', 'r') as f:
    categories = json.loads(f.read())
    categories = list(map(str.capitalize, categories))

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    
    try:
        # List of images encoded in base64
        img_list = data.get("images")
    except KeyError:
        return "Missing parameter 'images'", 400
    
    if not type(img_list) == list:
        return "Bad format: 'images' parameter must be a list", 400
    
    # Holds all images in np array format with shape (160, 160)
    img_batch = []
    
    # Convert images from encoded base64 to np array format 
    for img_data in img_list:
        try:
            img_base64 = img_data.split(",")[1]
        except AttributeError:
            return f"Entry in 'images' list is not a base64 image: '{img_data[:5]}'", 400
        
        decoded_img = b64decode(img_base64)
        try:
            img = Image.open(BytesIO(decoded_img))
        except UnidentifiedImageError:
            return "Unsupported file extension. Use .jpg or .png", 400 
            
        if img.mode == 'RGBA':
            img = img.convert("RGB")
        elif img.mode == 'CMYK':
            img = img.convert('CMYK')
        # elif not img.mode == 'RGB':
            return f"Unsupported color mode {img.mode}. Accepted: RGB, RGBA and CMYK", 400
        
        # Smart resize crops and resizes as to maintain the original image's aspect ratio
        resized_img = tf.keras.preprocessing.image.smart_resize(img, ((160, 160)))
        img_arr = tf.keras.utils.img_to_array(resized_img)
        if len(img_arr.shape) == 2:
            return f"Image has a single color channel. Expected RGB.", 400
        
        img_batch.append(img_arr)
        
    img_batch = np.array(img_batch)
        
    # Convert numpy array batch to tensor to feed to model
    input_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    predictions = model.predict(input_batch, verbose=False)
    
    image_tags = []
    for prediction in predictions:
        # Choose the indices of the top 3 predictions
        top2_pred_indices = np.argsort(prediction)[::-1][:2]
        
        # Create a mapping from category to prediction probability for each top 3 result
        top2_pred = { categories[idx]: prediction[idx] for idx in top2_pred_indices }
        
        # Collection of (category, probability) pairs
        top2_pairs = list(top2_pred.items())
        
        # Formula for picking a "good" set of image tags    
        tags = []
        
        # If top tag has 90% probability, select it only
        if top2_pairs[0][1] > 0.9:
            tags.append(top2_pairs[0][0])
        # When top 2 sum to at least 50%, choose both
        elif top2_pairs[0][1] + top2_pairs[1][1] > 0.5:
            tags.append(top2_pairs[0][0])
            tags.append(top2_pairs[1][0])
        else:
            tags.append("Unknown")
        
        image_tags.append(tags)
    print(f'{image_tags=}')
    return { "tags": image_tags }, 200

if __name__ == '__main__':
    app.run(debug=True)