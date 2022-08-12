import tensorflow as tf
keras = tf.keras
import numpy as np
from PIL import Image

import os # Ignore all debugging information
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, json

model = keras.models.load_model("./models/loss__0.33__acc__0.74")

# load category mappings
mapping_fp = open("./dataset/categories.json", "r")
categories = np.array(json.loads(mapping_fp.read()))
mapping_fp.close()

def main():
    
    try:
        img_path = sys.argv[1]
    except IndexError:
        print(
            """Usage: 'py classify.py <path_to_image>'
            Image must be in RGB format."""
            )
        return

    try:
        img = Image.open(img_path)
    except FileNotFoundError as e:
        print(f"Error: {e.strerror}.")
        return
    
    resized_img = keras.preprocessing.image.smart_resize(img, ((160, 160)))
    input_arr = keras.utils.img_to_array(resized_img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    # Predict the image scene
    prediction = model.predict(input_arr, verbose=False)

    sorted_indices = np.argsort(prediction[0])[::-1][:3]
    top_3_pred = { categories[idx]: prediction[0][idx] for idx in sorted_indices }
    
    print(f"Image Scene prediction: {top_3_pred}.")
    
if __name__ == '__main__':
    main()