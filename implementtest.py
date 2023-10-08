import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
# Load the trained model
imgpth=r"Path_to_test_image"
model = tf.keras.models.load_model(r"path_to_model")

# Function to load and preprocess an input image
def load_and_preprocess_image(image_path, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Specify the path to your input image
input_image_path = imgpth

# Load and preprocess the input image
input_image = load_and_preprocess_image(input_image_path)

# Make predictions
predictions = model.predict(input_image)
print(predictions)

# Convert predictions to binary form
prediction_label = "Fire" if predictions[0][0] > 0.5 else "no Fire"



# # Display the input image and predictions
# img = image.load_img(input_image_path, target_size=(150, 150))
# plt.imshow(img)
# plt.title(f'Prediction: {prediction_label}')
# plt.show()
# import cv2
# import numpy as np

def identify_fire(input_image_path):
    # Read the input image
    img = cv2.imread(input_image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the color red (fire color in HSV)
    lower_red = np.array([10, 191, 193])
    upper_red = np.array([30, 234, 255])

    # Create a mask to threshold the image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all contours into one bounding box
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        x = min(x, x_)
        y = min(y, y_)
        w = max(w, x_ + w_) - x
        h = max(h, y_ + h_) - y

    # Draw one bounding box around all detected fire regions
    cv2.rectangle(img, (x+30, y+30), (x + w-30, y + h-30), (0, 255, 0), 2)
    cv2.putText(img, f'Prediction: {prediction_label}', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Display the result
    down_width = 700
    down_height = 700
    down_points = (down_width, down_height)
    resized_down = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Fire Detection', resized_down)
    cv2.resizeWindow("Fire Detection", 700, 700)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your input image
input_image_path = imgpth

# Call the function to identify fire and display the result
identify_fire(input_image_path)
