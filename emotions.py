# Hey team, this is our emotion detection script! Let's break it down.
import numpy as np  # For handling arrays (like image data)
import argparse  # To handle command-line arguments like --mode
import matplotlib.pyplot as plt  # For plotting training results
import cv2  # For webcam and image processing
from tensorflow import keras  # TensorFlow's Keras for building our model
from keras.models import Sequential  # To build our model layer by layer
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization  # Layers for our neural network
from keras.optimizers import Adam  # Our optimizer for training
from keras.preprocessing.image import ImageDataGenerator  # For loading and augmenting images
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # To make training smarter
import os  # For handling file paths
import kagglehub  # To download the FER-2013 dataset
import zipfile  # To unzip the dataset
import shutil  # To clean up directories

# Let's keep TensorFlow quiet so we don't get too many logs (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This function grabs the FER-2013 dataset from Kaggle and sets it up for us
def setup_dataset():
    """Download and extract FER-2013 dataset from Kaggle."""
    try:
        print("Downloading the FER-2013 dataset... might take a sec!")
        path = kagglehub.dataset_download("msambare/fer2013")
        print("Path to dataset files:", path)

        # If it's a zip file, we need to extract it
        extract_dir = './fer2013_data'
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)  # Clean up any old data
        if path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
        else:
            extract_dir = path

        # Set up the paths for training and validation data
        train_dir = os.path.join(extract_dir, 'train')
        val_dir = os.path.join(extract_dir, 'test')
        
        # Double-check that the directories exist
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError("Train or test directory not found in extracted dataset.")
        
        print("Train dir contents:", os.listdir(train_dir))
        print("Test dir contents:", os.listdir(val_dir))
        return train_dir, val_dir
    except Exception as e:
        print(f"Oops, something went wrong with the dataset setup: {e}")
        exit(1)

# This function plots how our model did during training (accuracy and loss)
def plot_model_history(model_history):
    """Plot accuracy and loss curves given the model_history."""
    print("Let's plot how our training went!")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Plot accuracy (how good our model is at predicting emotions)
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    # Plot loss (how much our model is "wrong" during training)
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    # Save the plot so we can check it out later
    fig.savefig('/app/output/plot.png')
    plt.show()

# Here's where we build our neural network model for detecting emotions
def build_model():
    """Build and return an improved CNN model for emotion detection."""
    print("Building our emotion detection model... let's make it awesome!")
    model = Sequential()
    # Block 1: First set of layers to start learning basic features
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())  # Helps the model train faster and better
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Shrinks the image to focus on important parts
    model.add(Dropout(0.3))  # Randomly turns off some neurons to prevent overfitting

    # Block 2: More layers to learn more complex features
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Block 3: Even more layers for deeper learning
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Now we flatten the data and add dense layers to make predictions
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  # A big layer to combine all the features
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # More dropout to keep things balanced
    model.add(Dense(7, activation='softmax'))  # Final layer: 7 emotions to predict

    return model

# Let's get to the main part of the script!
if __name__ == "__main__":
    # First, let's see what mode we're running in (train or display)
    parser = argparse.ArgumentParser(description="Emotion Detection Script")
    parser.add_argument("--mode", type=str, default="train", help="Mode: 'train' or 'display'")
    args = parser.parse_args()
    mode = args.mode.lower()

    # Make sure the mode is valid
    if mode not in ["train", "display"]:
        print("Oops! Invalid mode. Use 'train' or 'display'.")
        exit(1)

    # Set up the dataset (download and extract FER-2013)
    train_dir, val_dir = setup_dataset()

    # Let's set up how we'll load the images for training and validation
    batch_size = 64  # How many images we process at once
    num_epoch = 100  # How many times we go through the dataset (we can increase this later)
    num_train = 28709  # Number of training images in FER-2013
    num_val = 7178     # Number of validation images in FER-2013

    # Add some cool transformations to the training images to make our model better
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize the pixel values
        rotation_range=30,  # Rotate images a bit
        width_shift_range=0.2,  # Shift images left/right
        height_shift_range=0.2,  # Shift images up/down
        shear_range=0.2,  # Add some shear (like tilting)
        zoom_range=0.2,   # Zoom in a bit
        horizontal_flip=True,  # Flip images horizontally
        fill_mode='nearest'  # Fill in any gaps
    )
    # For validation, we just normalize the images (no fancy transformations)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),  # Resize images to 48x48 (FER-2013 standard)
        batch_size=batch_size,
        color_mode="grayscale",  # Images are grayscale
        class_mode='categorical'  # Labels are one-hot encoded (7 emotions)
    )

    # Load the validation images
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )

    # Just to make sure we loaded the right number of images
    print(f"Found {train_generator.samples} training images.")
    print(f"Found {validation_generator.samples} validation images.")

    # Build our model
    model = build_model()

    if mode == "train":
        # Let's set up how we want to train the model
        initial_learning_rate = 0.003  # Starting learning rate (we'll adjust it if needed)
        optimizer = Adam(learning_rate=initial_learning_rate)
        model.compile(loss='categorical_crossentropy',  # Loss function for classification
                      optimizer=optimizer,
                      metrics=['accuracy'])  # We want to track accuracy

        # Add some smart features to make training better
        early_stopping = EarlyStopping(
            monitor='val_accuracy',  # Stop if validation accuracy stops improving
            patience=10,  # Wait 10 epochs before stopping
            restore_best_weights=True  # Keep the best weights
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',  # Reduce learning rate if validation loss stops improving
            factor=0.5,  # Cut the learning rate in half
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-6  # Don't go below this learning rate
        )
        checkpoint = ModelCheckpoint(
            '/app/output/best_model.h5',  # Save the best model here
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        # Time to train the model!
        print("Starting training... this might take a while, grab a snack!")
        try:
            model_info = model.fit(
                train_generator,
                steps_per_epoch=num_train // batch_size,  # How many batches per epoch
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_val // batch_size,
                callbacks=[early_stopping, reduce_lr, checkpoint]  # Use our smart features
            )
            plot_model_history(model_info)  # Plot how the training went
            model.save_weights('/app/output/model.h5')  # Save the final weights
            print("Training completed and weights saved to '/app/output/model.h5'.")
            print("Best model weights saved to '/app/output/best_model.h5'.")
        except Exception as e:
            print(f"Oh no, training failed: {e}")

    elif mode == "display":
        # Let's use the model to detect emotions in real-time!
        # First, let's check where we're looking for the model weights
        current_dir = os.getcwd()
        weights_path = os.path.join(current_dir, 'output', 'best_model.h5')
        print("Current working directory:", current_dir)
        print("Model weights path:", weights_path)
        print("Does best_model.h5 exist?", os.path.exists(weights_path))

        # Load the weights so we can make predictions
        try:
            model.load_weights(weights_path)  # Load the best weights
            print("Model weights loaded successfully. Let's detect some emotions!")
        except FileNotFoundError:
            print("Model weights not found at:", weights_path)
            print("Please ensure the model weights are in the 'output' directory.")
            exit(1)

        # Turn off OpenCL to avoid any OpenCV issues
        cv2.ocl.setUseOpenCL(False)

        # Map the model's predictions to emotion labels
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                        4: "Neutral", 5: "Sad", 6: "Surprised"}

        # Open the webcam for real-time detection
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Make sure it's connected!")
            exit(1)

        # Load the face detection model (Haar cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Could not load Haar cascade file. Ensure OpenCV is installed correctly.")
            exit(1)

        print("Starting real-time emotion detection... press 'q' to quit!")
        while True:
            ret, frame = cap.read()  # Grab a frame from the webcam
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Loop through each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                # Get the face region for emotion prediction
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                # Predict the emotion
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                # Show the emotion label above the face
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame with the predictions
            cv2.imshow('Emotion Detection', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped real-time detection.")