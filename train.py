import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from google.colab import drive

from preprocess import load_data

IMG_SIZE = 70

data, labels = load_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name="engagement_model"):
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Load pre-trained VGG16 model (excluding top layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Build the custom model
    model = Sequential()
    model.add(base_model)  # Add pre-trained VGG16 model
    model.add(Flatten())  # Flatten the output from the VGG16 base model
    model.add(Dense(512, activation='relu'))  # Add a dense layer with more units
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model with a learning rate scheduler
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Early stopping with more patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with data augmentation
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=10,  # Increased number of epochs
              validation_data=(X_val, y_val),
              callbacks=[early_stopping, lr_scheduler])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save model to Google Drive
    model_path = f'/{model_name}.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Generate predictions and confusion matrix
    y_pred = model.predict(X_test).flatten()  # Predictions as probabilities
    y_pred_classes = np.round(y_pred).astype(int)  # Convert to binary class (0 or 1)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=["Not Engaged", "Engaged"], labels=[0, 1]))

    # Generate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Call the function with your data
train_and_evaluate_model(X_train, y_train, X_test, y_test)
