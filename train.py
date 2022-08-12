import tensorflow as tf
keras = tf.keras

from model import build_transfer_model

training_dir = "./dataset/training/"
models_dir = "./models/"

epochs = 25
validation_split = 0.15

img_shape = (160, 160, 3)
batch_size = 128
num_classes = 36
seed = 889

train_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    image_size=img_shape[:-1],
    subset="training",
    validation_split=validation_split,
    seed=seed,
    label_mode='categorical',
    crop_to_aspect_ratio=True
)

validation_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    image_size=img_shape[:-1],
    subset="validation",
    validation_split=validation_split,
    seed=seed,
    label_mode='categorical',
    crop_to_aspect_ratio=True
)


model = build_transfer_model(img_shape, num_classes)
model.summary()

history: tf.keras.callbacks.History = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs
)

final_loss = history.history['loss'][-1]
final_acc = history.history['val_categorical_accuracy'][-1]

model.save(f"{models_dir}loss__{final_loss:.2f}__acc__{final_acc:.2f}")

# A note on model accuracy
# Some classes are purposefully chosen to overlap with others as
# it is intended to select the "Top 3" predictions from the softmax
# distribution