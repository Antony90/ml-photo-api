import tensorflow as tf

keras = tf.keras

num_classes = 30

def build_transfer_model(img_shape):
    # Load the pretrined base model
    base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    
    inputs = keras.Input(shape=img_shape)
    
    # Map pixel values from [0, 255] to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dense(256, activation='relu')(x)
    
    # Dropout to lessen overfitting
    x = keras.layers.Dropout(0.4)(x)
    
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
    
    return model