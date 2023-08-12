# Image Scene Classifier

## Summary 

Keras model to classify images by 36 Camera Scenes typically found in mobile gallery apps, used for image tagging. This is inspired by the [Mobile AI 2021 Real-Time Camera Scene Detection Challenge](https://competitions.codalab.org/competitions/28113).
I created my own dataset of 9445 images, using Google Image Search results for each category.

The result is a REST API to tag images in batch. I used this in my AI-assisted [Smart Gallery app](https://github.com/Antony90/smart-gallery/) to
automatically tag user submitted images.


## API Usage

`uvicorn api.server:app` to start FastAPI REST server. Tested with python 3.9 and 3.11.

See https://localhost/docs for API documentation.

### Batch Classify Images

| Description | Get a list of scene classifications for a batch of images provided |
|-------------|--------------------------------------------------------------------|
| Endpoint    | `/classify/`                                                       |
| HTTP Method | `POST`                                                             |
| Request data| JSON string - Array of Base 64 encoded images                      |
|Response data| JSON string - Array of tag arrays corresponding to request data order. Tags are capitalized|

### Example

```js
client.post("/classify", {
    images: [
        "YXNrZGpoZm9pcC...",
        "0pcTNuNHB2dALa..."
        // ...
    ]
}).then(({ data }) => {
    // console.log(data);
    {
        tags: [
            ['Nature', 'Coast'],
            ['Portrait', 'Family'],
            // ...
        ]
    }
})
```

### Formula for selecting multiple tags

Since the keras model uses softmax activation on the output layer, each node's value is the probability of it falling under 
one of the image scene classes. The highest probability node represents the model's most confident prediction. 

We can look for more than one tag when the highest probability falls below 90%. When there is no significant difference
between the top few output probabilities, the tag `Unknown` is assigned.

```py
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
```

### Errors

| Description                                   | Error Code |
|-----------------------------------------------|------------|
| Missing parameter 'images'                    | 400        |
| Bad format: 'images' parameter must be a list | 400        |
| Entry in 'images' list is not a base64 image  | 400        |
| Unsupported file extension. Use .jpg or .png  | 400        |
| Unsupported color mode. Accepted: RGB, RGBA and CMYK | 400 |
---
## Script Usage

`py ./dataset/gen_dataset.py` to generate the images for the dataset provided `./dataset/chromedriver.exe` exists.

`py train.py` to train a new model.

`py classify.py <path_to_img>` to get the top 3 class predictions.

---

### Dataset creation
Use selenium to query the category name and download the image results. Preview images were used instead of the source image for their lower resolution.

Images are classified into the following 36 classes:
<table>
<tr>
</tr>
<tr>
<td>
<ul>
<li>Animals</li>
<li>Airplane</li>
<li>Baby</li>
<li>Beach</li>
<li>Bike</li>
<li>Bird</li>
<li>Bridge</li>
<li>Building</li>
<li>Cake</li>
</ul>
</td>

<td>
<ul>
<li>Car</li>
<li>Cat</li>
<li>Child</li>
<li>Coast</li>
<li>Dance</li>
<li>Dog</li>
<li>Family</li>
<li>Flower</li>
<li>Food</li>
</ul>
</td>

<td>
<ul>
<li>Forest</li>
<li>Fruit</li>
<li>Holiday</li>
<li>House</li>
<li>Lake</li>
<li>Landmark</li>
<li>Meme</li>
<li>Mountain</li>
<li>Nature</li>
</ul>
</td>

<td>
<ul>
<li>Night</li>
<li>Painting</li>
<li>Portrait</li>
<li>Road</li>
<li>Sky</li>
<li>Snow</li>
<li>Sports</li>
<li>Sunset</li>
<li>Text document</li>
</ul>
</td>
</tr>
</table>



---
## Model Architecture

I implemented transfer learning using [MobileNetV2](https://arxiv.org/abs/1801.04381v4) for the base model. Input shape: `(160, 160, 3)`. Output shape: `(36, 1)`.


Since models have varying size, `keras.preprocessing.image.smart_resize( img, (160, 160) )` can be used to crop while maintaining 
the original image's aspect ratio.

```py
# Load the pretrined base model
base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
inputs = keras.Input(shape=img_shape)

# Map pixel values from [0, 255] to [-1, 1]
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x)

x = keras.layers.GlobalAveragePooling2D()(x)

# Dropout to lessen overfitting
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)

output = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=output)

# lock all MobileNetV2 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=keras.metrics.CategoricalAccuracy()
) 
```