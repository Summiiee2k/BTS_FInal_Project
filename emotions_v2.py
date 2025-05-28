import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

# Keep TensorFlow logs minimal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():
    """Load the custom DataFrame from the .pkl file."""
    file_path = 'A:/FINALPROJECT/main/balanced_df_with_images.pkl'
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_pickle(file_path)
        print(f"Loaded {len(df)} samples.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

def preprocess_dataframe(df):
    """Prepare the DataFrame by ensuring emotion_category is used directly."""
    print("Preprocessing DataFrame...")
    # Use emotion_category directly without remapping
    print("Emotion category distribution:")
    print(df['emotion_category'].value_counts())
    return df

def compute_class_weights(df, label_to_idx):
    """Compute class weights to handle class imbalance."""
    print("Checking class distribution to handle imbalance...")
    class_counts = Counter(df['emotion_category'])
    print("Class distribution:", dict(class_counts))
    total_samples = len(df)
    num_classes = len(class_counts)
    class_weights = {}
    for emotion, count in class_counts.items():
        class_weights[label_to_idx[emotion]] = (1 / count) * (total_samples / num_classes)
    print("Class weights:", class_weights)
    return class_weights

def build_model(num_classes):
    """Build a transfer learning model using MobileNetV2 with fine-tuning."""
    print("Building improved model with transfer learning (MobileNetV2)...")
    input_tensor = Input(shape=(96, 96, 3))
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

def custom_generator(df, batch_size, target_size, label_to_idx, mode='train'):
    """Custom generator to yield batches of preprocessed images and labels."""
    num_samples = len(df)
    indices = np.arange(num_samples)
    if mode == 'train':
        np.random.shuffle(indices)
    
    i = 0
    while True:
        batch_images = []
        batch_labels = []
        
        for _ in range(batch_size):
            if i >= num_samples:
                i = 0
                if mode == 'train':
                    np.random.shuffle(indices)
            
            idx = indices[i]
            row = df.iloc[idx]
            
            # Convert PIL Image to numpy array
            img = np.array(row['image'])
            # Convert grayscale to RGB
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # Resize to target size
            img = cv2.resize(img, target_size)
            # Normalize
            img = img.astype('float32') / 255.0
            
            # Data augmentation for training
            if mode == 'train':
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-40, 40)
                    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            
            batch_images.append(img)
            label = label_to_idx[row['emotion_category']]
            batch_labels.append(keras.utils.to_categorical(label, num_classes=len(label_to_idx)))
            
            i += 1
        
        yield np.array(batch_images), np.array(batch_labels)

def plot_model_history(history):
    """Plot training history for accuracy and loss."""
    print("Plotting training history...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history['accuracy'], label='train')
    axs[0].plot(history.history['val_accuracy'], label='val')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    axs[1].plot(history.history['loss'], label='train')
    axs[1].plot(history.history['val_loss'], label='val')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    fig.savefig('./output/plot.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Emotion Detection Script")
    parser.add_argument("--mode", type=str, default="train", help="Mode: 'train', 'evaluate', or 'display'")
    args = parser.parse_args()
    mode = args.mode.lower()

    if mode not in ["train", "evaluate", "display"]:
        print("Invalid mode. Please use 'train', 'evaluate', or 'display'.")
        exit(1)
    
    # Load and preprocess custom dataset
    df = load_data()
    df = preprocess_dataframe(df)

    # Dynamically create label mapping based on unique emotion categories
    unique_emotions = sorted(df['emotion_category'].unique())
    label_to_idx = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    idx_to_label = {idx: emotion for emotion, idx in label_to_idx.items()}
    num_classes = len(unique_emotions)
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_to_idx}")

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion_category'])
    num_train = len(train_df)
    num_val = len(val_df)
    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_val}")

    # Parameters
    batch_size = 32
    num_epoch = 200
    target_size = (96, 96)

    # Create custom generators
    train_generator = custom_generator(train_df, batch_size, target_size, label_to_idx, mode='train')
    validation_generator = custom_generator(val_df, batch_size, target_size, label_to_idx, mode='val')

    model = build_model(num_classes)

    if mode == "train":
        class_weights = compute_class_weights(train_df, label_to_idx)

        initial_learning_rate = 1e-4
        optimizer = Adam(learning_rate=initial_learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        checkpoint = ModelCheckpoint('./output/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
        
        print("Starting training... this may take a while.")
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=num_train // batch_size,
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_val // batch_size,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                class_weight=class_weights
            )
            plot_model_history(history)
            model.save_weights('./output/model.h5')
            print("Training completed. Final weights saved to './output/model.h5'.")
            print("Best model weights saved to './output/best_model.h5'.")
        except Exception as e:
            print(f"Training failed: {e}")

    elif mode == "evaluate":
        weights_path = os.path.join(os.getcwd(), 'output', 'best_model.h5')
        if not os.path.exists(weights_path):
            print("Model weights not found at:", weights_path)
            exit(1)
        model.load_weights(weights_path)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        print("Evaluating overall accuracy...")
        val_loss, val_accuracy = model.evaluate(validation_generator, steps=num_val // batch_size, verbose=1)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        print("Getting per-class performance...")
        validation_generator = custom_generator(val_df, batch_size, target_size, label_to_idx, mode='val')
        y_pred = model.predict(validation_generator, steps=num_val // batch_size, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = val_df['emotion_category'].map(label_to_idx).values[:len(y_pred_classes)]
        emotion_labels = list(label_to_idx.keys())
        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=emotion_labels))

    elif mode == "display":
        weights_path = os.path.join(os.getcwd(), 'output', 'best_model.h5')
        if not os.path.exists(weights_path):
            print("Best model weights not found. Ensure the weights are in the 'output' directory.")
            exit(1)
        try:
            model.load_weights(weights_path)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            exit(1)

        cv2.ocl.setUseOpenCL(False)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not accessible. Check your connection!")
            exit(1)

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
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                cropped_img = cv2.resize(roi_rgb, target_size)
                cropped_img = cropped_img.astype("float32") / 255.0
                cropped_img = np.expand_dims(cropped_img, axis=0)
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                emotion = idx_to_label[maxindex]
                cv2.putText(frame, emotion, (x+20, y-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")