from turtle import color
import tensorflow as tf
keras = tf.keras

import os
import numpy as np
from PIL import Image
test_dir = "./dataset/training/1_Portrait"
models_dir = "models"

model = keras.models.load_model("./models/loss__0.39")

for filename in os.listdir(test_dir)[5:]:
    img_path = os.path.join(test_dir, filename)
    
    img = Image.open(img_path)
    resized_img = img.resize((224, 224))
    resized_img.show()
    input_arr = keras.utils.img_to_array(resized_img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    
    print(prediction)
    print(np.argmax(prediction[0]))
    break