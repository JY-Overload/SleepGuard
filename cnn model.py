import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



base_dir = 'F:eye'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Directory with our training open eye pictures
train_openeye_dir = os.path.join(train_dir, 'open')

# Directory with our training close eye pictures
train_closeeye_dir = os.path.join(train_dir, 'close')

# Directory with our validation open eye pictures
validation_openeye_dir = os.path.join(validation_dir, 'open')

# Directory with our validation close eye pictures
validation_closeeye_dir = os.path.join(validation_dir, 'close')

# Directory with our test open and close eye pictures
test_openeeye_dir = os.path.join(test_dir, 'open')
test_closeeye_dir = os.path.join(test_dir, 'close')

# CNN model one: 2xConv layers + 2xFC layers without data augumentation
model_one = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(3,3), activation='relu',input_shape=(70,70,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model_one.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,)
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255,)
      # rotation_range=40,
      # width_shift_range=0.2,
      # height_shift_range=0.2,
      # shear_range=0.2,
      # zoom_range=0.2,
      # fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(70, 70),
    batch_size=50,
    color_mode="grayscale",
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(70, 70),
    batch_size=50,
    color_mode="grayscale",
    class_mode='binary')

history_one = model_one.fit(
      train_generator,
      steps_per_epoch=477,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=136,
      verbose=2)

from keras.models import model_from_json
model_json = model_one.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_one.save_weights("model2.h5")
print("Saved model to disk")