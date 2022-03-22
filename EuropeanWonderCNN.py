import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of tensorflow errors
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

batchSize = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

trainData = tf.keras.utils.image_dataset_from_directory(
  "train_data",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batchSize)

valData = tf.keras.utils.image_dataset_from_directory(
  "train_data",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batchSize)

classNames = trainData.class_names

AUTOTUNE = tf.data.AUTOTUNE

trainDS = trainData.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valDS = valData.cache().prefetch(buffer_size=AUTOTUNE)

numOfClasses = len(classNames)

augData = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ]
)

model = Sequential([
  augData,
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(numOfClasses)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1) # stop when accuracy slackens

cp = tf.keras.callbacks.ModelCheckpoint(
    "CNN_model/highestValAccuracy.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

graph = model.fit(
  trainDS,
  validation_data=valDS,
  epochs=50,
  callbacks=[cp]
)

model.save('EuropeanWonderModel.h5')

trainAcc = graph.history['accuracy']
valAcc = graph.history['val_accuracy']

trainLoss = graph.history['loss']
valLoss = graph.history['val_loss']

epochsRange = range(len(trainAcc))

plt.figure(figsize=(8, 8))
axA=plt.subplot(1, 2, 1)
axA.plot(epochsRange, trainAcc, label='Training Accuracy', linewidth=2)
axA.plot(epochsRange, valAcc, label='Validation Accuracy', color='green')
axA.legend(loc='lower right')
plt.title('Training and Validation Data Accuracy')

axL=plt.subplot(1, 2, 2)
axL.plot(epochsRange, trainLoss, label='Training Loss', linewidth=2)
axL.plot(epochsRange, valLoss, label='Validation Loss', color='green')
axL.legend(loc='upper right')
plt.title('Training and Validation Data Loss')
plt.show()