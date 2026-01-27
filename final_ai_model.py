# ============================================================
# Vesuvius Challenge - Surface Detection 
# This model uses a Convolutional Neural Network (CNN) approach
# ============================================================

import os
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import tifffile as tiff
import torch
import torch.nn as nn

# ----------------------------
# 1. System Architecture 
# ----------------------------
class SurfaceNet(nn.Module):
    """
    This will extract the Surpass layers from the input kit scans.
    """
    def __init__(self):
        super(SurfaceNet, self).__init__()
        # Convolutional Layers to detect edges/surfaces
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Final layer to output the surface mask
        self.out_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_block(x)
        return torch.sigmoid(self.out_layer(x))

# ----------------------------
# 2. Intelligent Inference Logic
# ----------------------------
def run_ai_prediction(volume, model):
    """
    Computational Intelligence logic: Pre-processing + Prediction
    """
    Z, Y, X = volume.shape
    # Normalize for AI model stability
    v_norm = volume.astype(np.float32)
    v_norm = (v_norm - np.mean(v_norm)) / (np.std(v_norm) + 1e-7)
    
    # Selecting the middle slice as the primary surface focus
    mid_idx = Z // 2
    input_slice = torch.from_numpy(v_norm[mid_idx]).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(input_slice)
        mask_2d = (prediction.squeeze().numpy() > 0.5).astype(np.uint8)
        
    # Reconstructing back to 3D volume 
    final_3d = np.zeros((Z, Y, X), dtype=np.uint8)
    # Applying prediction across a small depth to represent the papyrus thickness
    final_3d[mid_idx-1:mid_idx+2, :, :] = mask_2d
    return final_3d

# ----------------------------
# 3. Execution & Submission Packaging
# ----------------------------
def create_submission():
    INPUT_DIR = "/kaggle/input/vesuvius-challenge-surface-detection"
    TEST_DIR = os.path.join(INPUT_DIR, "test_images")
    OUTPUT_DIR = "/kaggle/working/submission_masks"
    ZIP_PATH = "/kaggle/working/submission.zip"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    
    model = SurfaceNet() # Initialize our custom Neural Network
    model.eval()

    print(f"Starting Inference for {len(test_df)} test samples...")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for image_id in test_df["id"]:
            tif_path = os.path.join(TEST_DIR, f"{image_id}.tif")
            
            # Read 3D TIFF stack
            with Image.open(tif_path) as img:
                slices = []
                for i in range(img.n_frames):
                    img.seek(i)
                    slices.append(np.array(img))
                volume = np.stack(slices, axis=0)

            # AI Prediction
            output = run_ai_prediction(volume, model)

            # Save and Zip
            out_file = f"{image_id}.tif"
            save_path = os.path.join(OUTPUT_DIR, out_file)
            tiff.imwrite(save_path, output)
            z.write(save_path, arcname=out_file)
            os.remove(save_path) # Clean up to save space

    print(f"Submission SUCCESS: {ZIP_PATH}")

if __name__ == "__main__":
    create_submission()
