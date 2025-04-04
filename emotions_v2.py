"""
Improved Emotion Detection Script with Transfer Learning

This version uses MobileNetV2 (pre-trained on ImageNet) to improve feature extraction,
and upscales the FER-2013 images from 48x48 to 96x96. Images are read in RGB mode so
that the pre-trained network can leverage its learned features. Once the top layers are
added, we fine-tune on our 7-class emotion problem.
"""

import os
import argparse
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import kagglehub
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Keep TensorFlow logs minimal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_dataset():
    """Download and extract FER-2013 dataset from Kaggle."""
    try:
        print("Downloading the FER-2013 dataset...")
        path = kagglehub.dataset_download("msambare/fer2013")
        print("Dataset path:", path)

        extract_dir = './fer2013_data'
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
        else:
            extract_dir = path

        train_dir = os.path.join(extract_dir, 'train')
        val_dir = os.path.join(extract_dir, 'test')
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError("Train or test directory not found in extracted dataset.")
        
        print("Train dir contents:", os.listdir(train_dir))
        print("Test dir contents:", os.listdir(val_dir))
        return train_dir, val_dir
    except Exception as e:
        print(f"Dataset setup error: {e}")
        exit(1)

def plot_model_history(history):
    """Plot training history for accuracy and loss."""
    print("Plotting training history...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Accuracy plot
    axs[0].plot(history.history['accuracy'], label='train')
    axs[0].plot(history.history['val_accuracy'], label='val')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    # Loss plot
    axs[1].plot(history.history['loss'], label='train')
    axs[1].plot(history.history['val_loss'], label='val')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    fig.savefig('./output/plot.png')
    plt.show()

def build_model():
    """Build a transfer learning model using MobileNetV2 for improved emotion detection."""
    print("Building improved model with transfer learning (MobileNetV2)...")
    # Define input shape for upscaled images (96x96) and 3 color channels.
    input_tensor = Input(shape=(96, 96, 3))
    
    # Load MobileNetV2 as a fixed feature extractor
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    base_model.trainable = False  # Freeze the base model
    
    # Add new classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Emotion Detection Script")
    parser.add_argument("--mode", type=str, default="train", help="Mode: 'train' or 'display'")
    args = parser.parse_args()
    mode = args.mode.lower()

    if mode not in ["train", "display"]:
        print("Invalid mode. Please use 'train' or 'display'.")
        exit(1)
    
    # Setup dataset directories
    train_dir, val_dir = setup_dataset()

    # Parameters and image size adjustments for transfer learning:
    batch_size = 64
    num_epoch = 100
    # Adjust these if needed based on your dataset counts
    num_train = 28709  
    num_val = 7178     
    target_size = (96, 96)  # Upscale images for pre-trained network input
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Validation generator (only normalization)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Since MobileNetV2 expects RGB images, we load images in 'rgb' mode (even though FER is grayscale)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )
    
    print(f"Found {train_generator.samples} training images.")
    print(f"Found {validation_generator.samples} validation images.")
    
    model = build_model()

    if mode == "train":
        initial_learning_rate = 1e-3
        optimizer = Adam(learning_rate=initial_learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Callbacks for efficient training
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        checkpoint = ModelCheckpoint('./output/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
        
        print("Starting training... this may take a while.")
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=num_train // batch_size,
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_val // batch_size,
                callbacks=[early_stopping, reduce_lr, checkpoint]
            )
            plot_model_history(history)
            model.save_weights('./output/model.h5')
            print("Training completed. Final weights saved to './output/model.h5'.")
            print("Best model weights saved to './output/best_model.h5'.")
        except Exception as e:
            print(f"Training failed: {e}")

    elif mode == "display":
        # For real-time emotion detection using webcam
        current_dir = os.getcwd()
        weights_path = os.path.join(current_dir, 'output', 'best_model.h5')
        print("Current working directory:", current_dir)
        print("Model weights path:", weights_path)
        if not os.path.exists(weights_path):
            print("Best model weights not found. Ensure the weights are in the 'output' directory.")
            exit(1)
        try:
            model.load_weights(weights_path)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            exit(1)

        # Disable OpenCL for OpenCV compatibility
        cv2.ocl.setUseOpenCL(False)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                        4: "Neutral", 5: "Sad", 6: "Surprised"}

        # Start video capture for real-time detection
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not accessible. Check your connection!")
            exit(1)

        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Could not load Haar cascade file.")
            exit(1)

        print("Starting real-time emotion detection... press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                # Extract face region and preprocess:
                roi_gray = gray[y:y+h, x:x+w]
                # Convert grayscale face to RGB and resize to match input size
                roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                cropped_img = cv2.resize(roi_rgb, target_size)
                cropped_img = cropped_img.astype("float32") / 255.0
                cropped_img = np.expand_dims(cropped_img, axis=0)
                # Predict emotion
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")