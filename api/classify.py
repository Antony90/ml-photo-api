import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class ImageSceneClassifier:
    def __init__(self,
                 model_path='./models/loss__0.33__acc__0.74',
                 categories_path='./dataset/categories.json'):
        self.model = tf.keras.models.load_model(model_path)
        with open(categories_path, 'r') as f:
            categories = json.loads(f.read())
            self.categories = list(map(str.capitalize, categories))
    
    def predict(self, images: list[np.ndarray]):
        img_batch = self._process_batch(images)
        # Convert numpy array batch to tensor to feed to model
        input_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        return self.model.predict(input_batch, verbose=False)
    
    def _process_img(self, img):
        # Smart resize crops and resizes as to maintain the original image's aspect ratio
        resized_img = tf.keras.preprocessing.image.smart_resize(img, ((160, 160)))
        return tf.keras.utils.img_to_array(resized_img)
    
    def _process_batch(self, img_batch):
        return np.array(list(map(self._process_img, img_batch)))
    
    def tags_from_predictions(self, predictions):
        image_tags = []
        for prediction in predictions:
            # Choose the indices of the top 3 predictions
            top2_pred_indices = np.argsort(prediction)[::-1][:2]
            
            # Create a mapping from category to prediction probability for each top 3 result
            top2_pred = { self.categories[idx]: prediction[idx] for idx in top2_pred_indices }
            
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
        return image_tags