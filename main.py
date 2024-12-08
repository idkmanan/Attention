import os
import cv2
import numpy as np
import tensorflow as tf 



IMG_SIZE = 70  # Resize images for uniformity

# Prepare the image (resizing, normalization, and converting grayscale to RGB)
def prepare_image(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # Convert grayscale to RGB by repeating the single channel three times
    rgb_array = cv2.cvtColor(resized_array, cv2.COLOR_GRAY2RGB)
    return rgb_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0  # Normalize pixel values

# Model analysis function
def analyze_images(model, test_folder):
    times = []
    attentionspan = []
    start_index = None

    # Sort images by filename (useful for ensuring sequence order)
    image_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(test_folder, img_name)
        try:
            prediction = model.predict(prepare_image(img_path))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

        # Check if the model predicts the image as "engaged" (probability >= 0.5)
        if prediction[0][0] >= 0.5:
            if start_index is None:
                start_index = idx
        else:
            if start_index is not None:
                end_index = idx
                times.append((start_index, end_index))
                attentionspan.append(end_index - start_index)
                start_index = None

    # Calculate and display the average attention span
    average_attention_span = sum(attentionspan) / len(attentionspan) if attentionspan else 0
    print("Attention times (indexes):", times)
    print("Average Attention Span (in frames):", average_attention_span)
    print(f"Total periods of attention: {len(times)}")

# Load the model from the file
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)  # Load the model from the path
    return model

# Example usage
model_path = '/engagement_model.h5'  # Path to your saved model
test_images_path = '/your_test_images'  # Path to your test images

# Load the trained model
model = load_model(model_path)

# Analyze images with the trained model
analyze_images(model, test_images_path)
