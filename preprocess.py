import os
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split


# Categories based on your dataset structure
CATEGORIES = {
    "Engaged": ["confused", "engaged", "frustrated"],
    "Not_engaged": ["bored", "drowsy", "looking_away"]
}

IMG_SIZE = 70  # Resize images for uniformity
DATASET_PATH = '/Student-engagement-dataset'

def load_data():
    data = []
    labels = []

    for main_category, subcategories in CATEGORIES.items():
        for subcategory in subcategories:
            category_path = os.path.join(DATASET_PATH, main_category, subcategory)
            label = 1 if main_category == "Engaged" else 0

            if not os.path.exists(category_path):
                print(f"Warning: Path does not exist - {category_path}")
                continue

            image_count = 0
            for img_filename in os.listdir(category_path):
                img_path = os.path.join(category_path, img_filename)

                if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                try:
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        print(f"Warning: Could not read image - {img_path}")
                        continue

                    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    # Convert grayscale to RGB
                    rgb_array = cv2.cvtColor(resized_array, cv2.COLOR_GRAY2RGB)
                    data.append(rgb_array)
                    labels.append(label)
                    image_count += 1
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

            print(f"Processed {image_count} images in {category_path}")

    # Convert data to NumPy arrays and normalize pixel values
    data = np.array(data) / 255.0
    labels = np.array(labels)

    return data, labels


# Load data and split it into training and testing sets
data, labels = load_data()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# Optionally print some information
print(f"Data shape: {data.shape}")
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
