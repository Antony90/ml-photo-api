import tensorflow as tf
keras = tf.keras

from model import build_transfer_model

training_dir = "./dataset/training/"
testing_dir = "./dataset/testing/"
models_dir = "./models/"

epochs = 60
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

# # https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets/58452268#58452268
# # Use 10% of the dataset for testing the final trained model

# is_test = lambda a, _: a % 10 == 0
# is_train = lambda a, b: not is_test(a, b)

# recover = lambda _, b: b # revert from "enumerate" form to original

# import numpy as np

# train_ds = np.array(all_ds.enumerate().filter(is_train).map(recover))
# test_ds = np.array(all_ds.enumerate().filter(is_test).map(recover))

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
