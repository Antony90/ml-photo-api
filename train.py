import tensorflow as tf
keras = tf.keras

from model import build_transfer_model

training_dir = "./dataset/training/"
testing_dir = "./dataset/testing/"
models_dir = "./models/"

epochs = 10
validation_split = 0.15

img_shape = (224, 224, 3)
batch_size = 128

train_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    image_size=img_shape[:-1],
    subset="training",
    validation_split=validation_split,
    seed=555,
    label_mode='categorical',
)

validation_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    image_size=img_shape[:-1],
    subset="validation",
    validation_split=validation_split,
    seed=555,
    label_mode='categorical',
)



model = build_transfer_model(img_shape)
model.summary()

history: tf.keras.callbacks.History = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs
)

final_loss = history.history['loss'][-1]
model.save(f"{models_dir}loss__{final_loss:.2f}")

# results = model.evaluate(test_ds)
# print(f"Test dataset [ loss, accuracy ]: {results}")
