import tensorflow as tf

keras = tf.keras

num_classes = 30

def build_transfer_model(img_shape):
    # Load the pretrined base model
    base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')

    x = base_model.output
    x = keras.layers.AveragePooling2D(pool_size=(4, 4), padding="valid")(x)
    x = keras.layers.Flatten()(x)
    
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    output = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(), 
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=keras.metrics.CategoricalAccuracy()
    )
    
    return model