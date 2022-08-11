# Image Scene Classifier

## Summary 

Keras model to classify images by Camera Scene. This follows from the competition [Mobile AI 2021 Real-Time Camera Scene Detection Challenge](https://competitions.codalab.org/competitions/28113).
<ul style="-webkit-column-count: 3; -moz-column-count: 3; column-count: 3;">
    <li>Dogs</li>
    <li>Group Portrait</li>
    <li>Portrait</li>
    <li>Beach</li>
    <li>Macro / Close-up</li>
    <li>Kids / Infants</li>
    <li>Food / Gourmet</li>
    <li>Mountains</li>
    <li>Architecture</li>
    <li>Snow</li>
    <li>Waterfall</li>
    <li>Greenery / Grass</li>
    <li>Cats</li>
    <li>Overcast / Cloudy Sky</li>
    <li>Landscape</li>
    <li>Sunrise / Sunset</li>
    <li>Underwater</li>
    <li>Blue Sky</li>
    <li>Flower</li>
    <li>Candle light</li>
    <li>Night</li>
    <li>Autumn Plants</li>
    <li>Shot Stage / Concert</li>
    <li>Monitor Screen</li>
    <li>Indoor</li>
    <li>Neon Lights / Neon Signs</li>
    <li>Text / Document</li>
    <li>Fireworks</li>
    <li>Backlight / Contre-jour</li>
    <li>QR Code</li>
</ul>

---
## Model Architecture

I implemented transfer learning using [MobileNetV2](https://arxiv.org/abs/1801.04381v4) for the base model.

```
# Load the pretrined base model

base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')

x = base_model.output
x = keras.layers.AveragePooling2D(pool_size=(4, 4), padding="valid")(x)
x = keras.layers.Flatten()(x)

output = keras.layers.Dense(256, activation='relu')(x)
output = keras.layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
```