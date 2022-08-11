import tensorflow as tf
keras = tf.keras
from PIL import Image
import numpy as np

import os
import json
from time import sleep

test_dir = "./dataset/testing"
models_dir = "./models/"

# load category mappings
mapping_fp = open("categories.json", "r")
categories = json.loads(mapping_fp.read())
mapping_fp.close()

model = keras.models.load_model(models_dir + "loss__0.06")

for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = Image.open(img_path)
    
    # Resize to model input shape
    resized_img = img.resize((224, 224))
    input_arr = keras.utils.img_to_array(resized_img)
    
    input_arr = np.array([input_arr])  # Convert single image to a batch
    
    try:
        prediction = model.predict(input_arr, verbose=False)
    except tf.errors.InvalidArgumentError:
        print(f"{filename} is greyscale, skipping")
        continue
    
    # Choose index with highest probability
    idx = np.argmax(prediction[0])
    print(categories[idx])
    img.show()
    sleep(2)