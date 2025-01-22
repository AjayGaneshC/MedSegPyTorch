import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json

# Import your project modules
from configs import MyConfig
from models import get_model

class PolyPredictorApp:
    def __init__(self):
        # Initialize configuration
        self.config = MyConfig()
        self.config.is_testing = True
        
        # Update model configuration for SMP
        self.config.model = 'smp'
        self.config.decoder = 'unet'  # Adjust based on your checkpoint
        self.config.encoder = 'resnet50'  # Adjust based on your checkpoint
        self.config.encoder_weights = 'imagenet'
        
        # Determine the number of classes from the checkpoint
        try:
            model_path = self.config.model_path
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract state dictionary
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
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
            # Extract state dictionary
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
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
            # Try to load colormap from config
            colormap_path = self.config.colormap_path if hasattr(self.config, 'colormap_path') else 'save/colormap.json'
            
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
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Apply transformations
        transformed = self.transform(image=image_np)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return input_tensor

    def predict(self, input_tensor):
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Process prediction
        if self.config.num_class > 1:
            # Multi-class segmentation
            pred_mask = torch.softmax(prediction, dim=1)
            pred_mask = pred_mask.argmax(dim=1).squeeze().cpu().numpy()
        else:
            # Binary segmentation
            pred_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        return pred_mask

    def visualize_prediction(self, original_image, mask):
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (original_image.width, original_image.height), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Create color overlay
        color_mask = np.zeros_like(np.array(original_image))
        
        if self.config.num_class > 1:
            # Multi-class visualization
            unique_classes = np.unique(mask_resized)
            for cls in unique_classes:
                if cls > 0:
                    # Get color from colormap, fallback to default if not found
                    color = self.colormap.get(str(cls), [255, 0, 0])
                    color_mask[mask_resized == cls] = color
        else:
            # Binary segmentation visualization
            color_mask[mask_resized == 1] = [255, 0, 0]  # Red color for positive class
        
        # Blend original image with mask
        blended = cv2.addWeighted(np.array(original_image), 0.7, color_mask, 0.3, 0)
        
        return blended

    def run(self):
        st.title('Medical Image Segmentation')
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a medical image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            try:
                # Preprocess and predict
                input_tensor = self.preprocess_image(image)
                mask = self.predict(input_tensor)
                
                # Visualize prediction
                blended_image = self.visualize_prediction(image, mask)
                
                # Display results
                st.subheader('Prediction Results')
                st.image(blended_image, caption='Segmentation Result', use_column_width=True)
                
                # Optional: Display mask separately
                st.image(mask * (255 // (self.config.num_class - 1)), 
                         caption='Segmentation Mask', 
                         use_column_width=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

def main():
    try:
        app = PolyPredictorApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")

if __name__ == '__main__':
    main()
