import tensorflow as tf
keras = tf.keras
from PIL import Image
import numpy as np

import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

test_dir = "./dataset/testing"
models_dir = "./models/"

# load category mappings
mapping_fp = open("./dataset/categories.json", "r")
categories = json.loads(mapping_fp.read())
categories = np.array(categories)
mapping_fp.close()

# "loss__0.02__acc__0.92"

model = keras.models.load_model(models_dir + "loss__0.33__acc__0.74")

for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = Image.open(img_path)
    
    # Resize and crop to model input shape
    resized_img = keras.preprocessing.image.smart_resize(img, (160, 160))
    input_arr = keras.utils.img_to_array(resized_img)
    
    input_arr = np.array([input_arr])  # Convert single image to a batch
    
    try:
        prediction = model.predict(input_arr, verbose=False)
    except tf.errors.InvalidArgumentError:
        print(f"{filename} is greyscale, skipping")
        continue
    
    # Choose index with highest probability
    sorted_indices = np.argsort(prediction[0])[::-1][:3]
    top_3_pred = { categories[idx]: prediction[0][idx] for idx in sorted_indices }
    
    print(top_3_pred)
    img.show()
    
    input("Waiting for confirmation")
    
    # Formula for picking a "good" set of predictions
    # if p[0] > 0.9: p[0]
    # if p[0] + p[1] > 0.5: p[0], p[1]
    # else Unknown  