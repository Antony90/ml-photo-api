import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load('scene_parse150')
print(ds)