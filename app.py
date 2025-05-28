import os
import pandas as pd
import numpy as np
import cv2

# Load the .pkl file
file_path = 'A:/FINALPROJECT/main/combined_data_balanced.pkl'
print(f"Loading data from {file_path}...")
df = pd.read_pickle(file_path)
print(f"Loaded {len(df)} samples.")

# Inspect DataFrame structure
print("\nDataFrame Columns:")
print(df.columns.tolist())
print("\nEmotion Category Distribution:")
print(df['emotion_category'].value_counts())

# Create output directory if it doesn't exist
output_dir = 'output/inspected_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Process the first 20 images
for i in range(min(20, len(df))):
    # Get the image (already a numpy array in 3x224x224 format)
    img = np.array(df['image'].iloc[i])
    
    # Verify original shape (should be 3x224x224)
    print(f"\nImage {i} original shape: {img.shape}")
    print(f"Image {i} dtype: {img.dtype}")
    print(f"Image {i} value range: Min {img.min():.4f}, Max {img.max():.4f}")
    
    # Transpose to (224, 224, 3) for visualization
    img = np.transpose(img, (1, 2, 0))
    print(f"Image {i} shape after transpose: {img.shape}")
    
    # Check the corresponding emotion category
    emotion = df['emotion_category'].iloc[i]
    print(f"Image {i} emotion category: {emotion}")
    
    # Since the image is already normalized (e.g., [-2.1, 2.64]), rescale to [0, 1] for visualization
    img_rescaled = (img - img.min()) / (img.max() - img.min())
    
    # Apply a slight red tint to visually confirm RGB format (increase red channel by 10%)
    img_tinted = img_rescaled.copy()
    img_tinted[:, :, 0] = img_tinted[:, :, 0] * 1.1  # Boost red channel (R in RGB)
    img_tinted = np.clip(img_tinted, 0, 1)  # Ensure values stay in [0, 1]
    
    # Convert to uint8 for saving
    img_for_save = (img_tinted * 255).astype('uint8')
    
    # Save the image
    output_path = os.path.join(output_dir, f'inspected_image_{i}_{emotion}.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(img_for_save, cv2.COLOR_RGB2BGR))
    print(f"Saved inspected image {i} to {output_path}")

print("\nInspection complete. Check the output/inspected_images/ directory for saved images.")