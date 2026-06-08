import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Dataset Paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='rgb',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# CNN Model
model = Sequential()

# Layer 1
model.add(Conv2D(
    32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=(48,48,3)
))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(
    64,
    kernel_size=(3,3),
    activation='relu'
))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
model.add(Conv2D(
    128,
    kernel_size=(3,3),
    activation='relu'
))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(256, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001
))
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train Model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=[early_stop]
)

# Save Model
model.save('models/emotion_model.h5')

print("Model Trained Successfully!")

# Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()