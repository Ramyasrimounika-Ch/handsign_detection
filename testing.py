import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("handsign_model.keras")

# Raw coordinates (x0, y0, x1, y1, ..., x20, y20)
raw_input = [574, 338, 514, 327, 457, 302, 413, 295, 374, 292, 480, 221, 450, 179, 422, 171, 397, 176, 495, 215, 463, 163, 430, 161, 406, 176, 509, 214, 477, 162, 440, 159, 413, 172, 521, 218, 487, 178, 455, 168, 430, 170]
print(raw_input[0])

# Assume standard webcam resolution
width = 640
height = 480

# Separate x and y
x_coords = raw_input[0::2]
y_coords = raw_input[1::2]

# Normalize
x_norm = [x / width for x in x_coords]
y_norm = [y / height for y in y_coords]

# Reconstruct input: [x0, x1, ..., x20, y0, y1, ..., y20]
normalized_input = x_norm + y_norm

# Final input to model
test_input = np.array([normalized_input])  # Shape: (1, 42)

# Predict
prediction = model.predict(test_input)
predicted_class = np.argmax(prediction)

print("Raw prediction output:", prediction)
print("Predicted class index:", predicted_class)
