import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of tensorflow errors
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

batch_size = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

train_data = tf.keras.utils.image_dataset_from_directory(
  "train_data",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batch_size)

val_data = tf.keras.utils.image_dataset_from_directory(
  "train_data",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batch_size)

class_names = train_data.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_data.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

augmented_data = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(IMG_HEIGHT,
                                    IMG_WIDTH,
                                    3)),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
  augmented_data,
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# callback = EarlyStopping(monitor='accuracy', patience=2) # stop when accuracy slackens

callback = tf.keras.callbacks.ModelCheckpoint(
    "CNN_model/highestValAccuracy.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

epochs=25
graph = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[callback]
)

model.save('WorldWonderModel.h5')

train_accuracy = graph.history['accuracy']
val_accuracy = graph.history['val_accuracy']

train_loss = graph.history['loss']
val_loss = graph.history['val_loss']

epochs_range = range(epochs)
# epochs_range = range(len(train_accuracy))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Data Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Data Loss')
plt.show()