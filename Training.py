# Importing the necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print(">> IMAGE CLASSIFICATION MODEL CREATION")
print(">> STEP#1 IMAGE PRE-PROCESSING")

# Resizing the Images to Preferred Size and applying Data Augmentation
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

test_image_generator = ImageDataGenerator(rescale=1.0 / 255)

# Reading the dataset directory
training_images = train_image_generator.flow_from_directory(
    'C:/Users/User/PycharmProjects/Pneumonia/Dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

testing_images = test_image_generator.flow_from_directory(
    'C:/Users/User/PycharmProjects/Pneumonia/Dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

print(">> STEP#2 CREATE CNN MODEL")
# Model Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))

print(">> STEP#3 TRAIN THE MODEL")
# Specify the learning rate
learning_rate = 0.0001
# Compiling the Model with specified learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model
history = model.fit(
    training_images,
    epochs=50,  # Increase the number of epochs
    validation_data=testing_images
)

print(">> STEP#4 VISUALIZING ACCURACY AND LOSS")
# Visualizing Accuracy and Loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the Model
model.save('PNEUMONIA_model')
print(">> MODEL SAVED ")

