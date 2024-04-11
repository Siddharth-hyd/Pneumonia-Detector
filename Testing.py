from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the pre-trained model
model = load_model("PNEUMONIA_model")

# No need to recompile the model if you're not changing its configuration or training it further
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess the test image
image = cv2.imread("C:/Users/User/Desktop/chest_xray/test/NORMAL/IM-0017-0001.jpeg")
image = cv2.resize(image, (64, 64))
image = image.astype('float32') / 255  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Prediction of the test image
prediction = model.predict(image)

# Get the predicted class label
if prediction[0][0] > 0.5:
    predicted_class = "PNEUMONIA"
else:
    predicted_class = "NORMAL"

# Print the predicted class
print("Predicted Class:", predicted_class)
