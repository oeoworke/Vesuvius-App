import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
import os


# 1. Model Architecture

class SurfaceNet(nn.Module):
    def __init__(self):
        super(SurfaceNet, self).__init__()
        # Convolutional layers - Detect edges and surface patterns
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.out_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # input x must be 4D: [Batch, Channel, Height, Width]
        x = self.conv_block(x)
        return torch.sigmoid(self.out_layer(x))


# 2. Page Configuration & UI
st.set_page_config(page_title="Vesuvius AI Surface Detector", layout="wide")

st.title("Vesuvius Challenge: Surface Detection AI App")
st.markdown("""
This application uses your **SurfaceNet AI model** to segment and display the 
papyrus surface from 3D CT scans.
""")

st.sidebar.header("Model Details")
st.sidebar.write("**Architecture:** CNN (SurfaceNet)")
st.sidebar.write("**Framework:** PyTorch")
st.sidebar.write("**Input:** .tif (Volumetric Slices)")


# 3. Model Loading Logic
@st.cache_resource
def load_trained_model():
    model = SurfaceNet()
    # Note: If no training weights (.pth) are provided, this uses random weights
    model.eval()
    return model

model = load_trained_model()

# 4. Image Upload & Prediction logic
uploaded_file = st.file_uploader("Upload a .tif slice...", type=["tif", "tiff"])

if uploaded_file is not None:
    with st.spinner('AI model is processing...'):
        # 1. Read the image using Tifffile
        raw_data = tiff.imread(uploaded_file)
        
        # 2. Dimension Handling 
        # If 3D volume (Depth, Height, Width), take only the middle slice
        if raw_data.ndim == 3:
            mid_idx = raw_data.shape[0] // 2
            working_image = raw_data[mid_idx]
        else:
            working_image = raw_data
            
        # 3. Pre-processing (Normalization)
        img_float = working_image.astype(np.float32)
        norm_img = (img_float - np.mean(img_float)) / (np.std(img_float) + 1e-7)
        
        # 4. AI Inference
        # Convert input to 4D: [1, 1, Height, Width]
        input_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            mask_2d = (prediction.squeeze().numpy() > 0.5).astype(np.uint8)

        # 5. Results Visualization
        st.success("Surface Detection Success!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original CT Scan Slice")
            fig1, ax1 = plt.subplots()
            ax1.imshow(working_image, cmap='gray')
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col2:
            st.subheader("AI Detected Surface (Overlay)")
            fig2, ax2 = plt.subplots()
            ax2.imshow(working_image, cmap='gray')
            ax2.imshow(mask_2d, cmap='jet', alpha=0.5) 
            ax2.axis('off')
            st.pyplot(fig2)

        st.info(f"Detected Surface Pixels: {np.sum(mask_2d)}")
        st.balloons()
else:
    st.warning("Please upload a .tif slice from the Vesuvius dataset to proceed.")

st.markdown("---")
st.caption("Developed By A.Anustan")