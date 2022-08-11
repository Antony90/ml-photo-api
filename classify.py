import tensorflow as tf
keras = tf.keras
import numpy as np

import os # Ignore all debugging information
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, json

model = keras.models.load_model("./models/loss__0.06")

# load category mappings
try:
    mapping_fp = open("categories.json", "r")
except FileNotFoundError:
    print("Category mappings do not exist, run `py gen_category_map.py`.")
else:
    category_mapping = json.loads(mapping_fp.read())
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
        img = keras.utils.load_img(img_path, target_size=(224,224))
    except FileNotFoundError as e:
        print(f"Error: {e.strerror}.")
        return
    
    input_arr = keras.utils.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    # Predict the image scene
    prediction_idx = np.argmax(model.predict(input_arr, verbose=False))
    
    category = category_mapping[prediction_idx]
    
    print(f"Image Scene prediction: '{category}'.")
    
if __name__ == '__main__':
    main()