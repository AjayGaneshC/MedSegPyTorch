import streamlit as st
import torch
import numpy as np
import cv2
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
import tempfile

# Import your project modules
from configs import MyConfig
from models import get_model

class PerformanceTracker:
    def __init__(self):
        self.inference_times = []
        self.preprocessing_times = []
        self.visualization_times = []

    def add_inference_time(self, time):
        self.inference_times.append(time)

    def add_preprocessing_time(self, time):
        self.preprocessing_times.append(time)

    def add_visualization_time(self, time):
        self.visualization_times.append(time)

    def get_metrics_dataframe(self):
        metrics = {
            'Metric': ['Average', 'Minimum', 'Maximum', 'Standard Deviation'],
            'Inference Time (ms)': [
                np.mean(self.inference_times) * 1000,
                np.min(self.inference_times) * 1000,
                np.max(self.inference_times) * 1000,
                np.std(self.inference_times) * 1000
            ],
            'Preprocessing Time (ms)': [
                np.mean(self.preprocessing_times) * 1000,
                np.min(self.preprocessing_times) * 1000,
                np.max(self.preprocessing_times) * 1000,
                np.std(self.preprocessing_times) * 1000
            ],
            'Visualization Time (ms)': [
                np.mean(self.visualization_times) * 1000,
                np.min(self.visualization_times) * 1000,
                np.max(self.visualization_times) * 1000,
                np.std(self.visualization_times) * 1000
            ]
        }
        return pd.DataFrame(metrics)

    def plot_time_distribution(self):
        # Create a figure with subplots for each time metric
        fig = go.Figure()
        
        # Inference Time Distribution
        fig.add_trace(go.Box(y=np.array(self.inference_times) * 1000, name='Inference Time (ms)'))
        
        # Preprocessing Time Distribution
        fig.add_trace(go.Box(y=np.array(self.preprocessing_times) * 1000, name='Preprocessing Time (ms)'))
        
        # Visualization Time Distribution
        fig.add_trace(go.Box(y=np.array(self.visualization_times) * 1000, name='Visualization Time (ms)'))
        
        fig.update_layout(
            title='Performance Metrics Distribution',
            yaxis_title='Time (milliseconds)',
            boxmode='group'
        )
        
        return fig

class PolyPredictorApp:
    def __init__(self):
        # Initialize configuration
        self.config = MyConfig()
        self.config.is_testing = True
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Update model configuration for SMP
        self.config.model = 'smp'
        self.config.decoder = 'unet'
        self.config.encoder = 'resnet50'
        self.config.encoder_weights = 'imagenet'
        
        # Initialize checkpoint as None
        self.checkpoint = None
        
        # Determine the number of classes from the checkpoint
        try:
            model_path = self.config.model_path
            self.checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract state dictionary
            if 'state_dict' in self.checkpoint:
                state_dict = self.checkpoint['state_dict']
            else:
                state_dict = self.checkpoint
            
            # Find the number of classes from the segmentation head
            segmentation_head_keys = [k for k in state_dict.keys() if 'segmentation_head.0.weight' in k or 'seg_head.weight' in k]
            
            if segmentation_head_keys:
                num_classes = state_dict[segmentation_head_keys[0]].shape[0]
                st.info(f"Detected {num_classes} classes from checkpoint")
            else:
                num_classes = 1  # Default to binary segmentation
                st.warning("Could not detect number of classes, defaulting to binary segmentation")
            
            # Update config with the correct number of classes
            self.config.num_channel = 3
            self.config.num_class = num_classes
            
        except Exception as e:
            st.warning(f"Error detecting classes: {e}")
            self.config.num_channel = 3
            self.config.num_class = 1  # Safe default
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = get_model(self.config).to(self.device)
        except Exception as e:
            st.error(f"Error creating model: {e}")
            raise
        
        # Load trained weights
        try:
            # Ensure checkpoint is loaded
            if self.checkpoint is None:
                model_path = self.config.model_path
                self.checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract state dictionary
            if 'state_dict' in self.checkpoint:
                state_dict = self.checkpoint['state_dict']
            else:
                state_dict = self.checkpoint
            
            # Remove keys that don't match the current model
            keys_to_remove = [k for k in list(state_dict.keys()) if k not in self.model.state_dict()]
            for k in keys_to_remove:
                del state_dict[k]
            
            # Load state dictionary
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            raise
        
        # Preprocessing transforms
        self.transform = A.Compose([
            A.Resize(320, 320),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Colormap handling
        try:
            # Ensure colormap_path is a string
            colormap_path = str(getattr(self.config, 'colormap_path', 'save/colormap.json'))
            
            if os.path.exists(colormap_path):
                with open(colormap_path, 'r') as f:
                    self.colormap = json.load(f)
            else:
                # Generate default colormap if file not found
                self.colormap = self.generate_default_colormap()
                st.warning(f"Colormap file not found at {colormap_path}. Using default colormap.")
        
        except Exception as e:
            st.warning(f"Error loading colormap: {e}")
            self.colormap = self.generate_default_colormap()

    def generate_default_colormap(self):
        """
        Generate a default colormap based on number of classes
        """
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Cyan
        ]
        
        # Return colors for each class
        return {str(i): list(color) for i, color in enumerate(colors[:self.config.num_class])}

    def preprocess_image(self, image):
        start_time = time.time()
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Apply transformations
        transformed = self.transform(image=image_np)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        preprocessing_time = time.time() - start_time
        self.performance_tracker.add_preprocessing_time(preprocessing_time)
        
        return input_tensor

    def process_image(self, image):
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            start_inference = time.time()
            output = self.model(input_tensor)
            inference_time = time.time() - start_inference
            self.performance_tracker.add_inference_time(inference_time)
        
        # Post-process output
        start_visualization = time.time()
        output_np = output.cpu().numpy()
        pred_mask = output_np[0].argmax(axis=0)
        
        # Colorize mask
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_idx in range(self.config.num_class):
            mask_class = pred_mask == class_idx
            colored_mask[mask_class] = self.colormap.get(str(class_idx), [0, 0, 0])
        
        visualization_time = time.time() - start_visualization
        self.performance_tracker.add_visualization_time(visualization_time)
        
        return colored_mask

    def process_image_multiple_sizes(self, image, sizes=[320, 480, 640, 800]):
        """
        Process the same image at multiple sizes and collect performance metrics
        
        Args:
            image (PIL.Image): Input image
            sizes (list): List of image sizes to resize and process
        
        Returns:
            dict: Dictionary containing results for each size
        """
        results = {}
        size_metrics = []
        
        # Ensure image is converted to RGB if it's not already
        image = image.convert('RGB')
        
        for size in sizes:
            # Create a new transform for each size
            size_transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            start_time = time.time()
            
            # Preprocess image
            image_np = np.array(image)
            transformed = size_transform(image=image_np)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            preprocessing_time = time.time() - start_time
            
            # Run inference
            with torch.no_grad():
                start_inference = time.time()
                output = self.model(input_tensor)
                inference_time = time.time() - start_inference
            
            # Post-process output
            start_visualization = time.time()
            output_np = output.cpu().numpy()
            pred_mask = output_np[0].argmax(axis=0)
            
            # Colorize mask
            colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            for class_idx in range(self.config.num_class):
                mask_class = pred_mask == class_idx
                colored_mask[mask_class] = self.colormap.get(str(class_idx), [0, 0, 0])
            
            visualization_time = time.time() - start_visualization
            
            # Track performance metrics
            size_metrics.append({
                'Size': f'{size}x{size}',
                'Preprocessing Time (ms)': preprocessing_time * 1000,
                'Inference Time (ms)': inference_time * 1000,
                'Visualization Time (ms)': visualization_time * 1000,
                'Total Time (ms)': (preprocessing_time + inference_time + visualization_time) * 1000
            })
            
            results[f'{size}x{size}'] = {
                'mask': colored_mask,
                'preprocessing_time': preprocessing_time,
                'inference_time': inference_time,
                'visualization_time': visualization_time
            }
        
        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame(size_metrics)
        
        return results, metrics_df

def main():
    st.title('Medical Image Segmentation')
    
    # Sidebar for configuration
    st.sidebar.header('Inference Configuration')
    
    # Initialize the app
    app = PolyPredictorApp()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a medical image...", type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader('Original Image')
        st.image(image, use_column_width=True)
        
        # Run inference on multiple sizes
        results, metrics_df = app.process_image_multiple_sizes(image)
        
        # Display results for each size
        st.subheader('Segmentation Results')
        
        # Create columns for images
        cols = st.columns(len(results))
        for i, (size, result) in enumerate(results.items()):
            with cols[i]:
                st.text(f'Size: {size}')
                st.image(result['mask'], use_column_width=True)
        
        # Display performance metrics
        st.subheader('Performance Metrics')
        st.dataframe(metrics_df)
        
        # Optional: Plotly visualization of metrics
        st.subheader('Performance Metrics Visualization')
        fig = go.Figure(data=[
            go.Bar(
                name='Preprocessing Time', 
                x=metrics_df['Size'], 
                y=metrics_df['Preprocessing Time (ms)']
            ),
            go.Bar(
                name='Inference Time', 
                x=metrics_df['Size'], 
                y=metrics_df['Inference Time (ms)']
            ),
            go.Bar(
                name='Visualization Time', 
                x=metrics_df['Size'], 
                y=metrics_df['Visualization Time (ms)']
            )
        ])
        fig.update_layout(barmode='group', title='Performance Metrics by Image Size')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
