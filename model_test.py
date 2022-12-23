import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

data_dir = pathlib.Path("images/Cyrillic")
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)
batch_size = 32
img_height = 32
img_width = 32

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = validation_dataset.class_names
print(class_names)


