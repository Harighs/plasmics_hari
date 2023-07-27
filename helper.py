#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.models.segmentation import deeplabv3_resnet50
import os
import cv2
import warnings
import numpy as np
from torchvision.transforms import ToPILImage
from skimage import exposure
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class Gaussian_Histogram():
    """
    This class is used to perform Gaussian Histogram for an given image.
    
    Methods:
    perform_gauss: This Method is used to perform Gaussian Histogram for an given image.
    
    helper_methods:
    gaussian: This Method is used to generate Gaussian filter.
    """
    def __init__(self):
        pass
    
    def perform_gauss(self, image, label):
        """
        This Method is used to perform Gaussian Histogram for an given image.
        
        Inputs:
            image: str or PIL.Image.Image (raises error if input is not str or PIL.Image.Image)
            label: str (label for the plot)
        
        Outputs:
            Prints out the Gaussian Histogram of the given image with pixel intensity on x-axis and frequency on y-axis.
        """
        # Sanity checks
        if isinstance(image, str):
            image = Image.open(image).convert("L")
            image_array = np.array(image)
        elif isinstance(image, Image.Image):
            image_array = np.array(image)

        if not isinstance(label, str):
            raise TypeError("label must be a string")
        
        # Calculate the histogram
        hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])
        mu = np.mean(image_array)
        sigma = np.std(image_array)
        
        # Generate Gaussian filter
        gaussian_filter = self.gaussian(bins, mu, sigma)
        
        # Plotting the results
        plt.figure(figsize=(5, 3))
        plt.hist(image_array.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.5, label='Histogram')
        plt.plot(bins, gaussian_filter * np.max(hist), color='red', label='Gaussian')
        plt.title(label)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
        return None
        
    @staticmethod
    def gaussian(x, mu, sigma):
        """
        Inputs:
            x: array of pixel intensity
            mu: mean of the pixel intensity
            sigma: standard deviation of the pixel intensity
            
        returns: 
            Calculated Gaussian filter
        """
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    
class Cat_Prediction():
    def __init__(self):
        """
        Cat Background Segmentation class using with ResNet50 backbone for PLASMICS.

        Attributes:
            model (torch.nn.Module): The pre-trained DeepLabV3 model with a ResNet50 backbone, loaded and ready for inference on GPU.
            preprocess (torchvision.transforms.Compose): Preprocessing pipeline for input images to be fed into the segmentation model.
        """
        self.model = deeplabv3_resnet50(pretrained=True).cuda()
        # Set the model to evaluation mode
        self.model.eval()
        self.preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        
    def segment(self, image):
        """
        Segment the cat from the background in the input image.

        Inputs:
            image (str or PIL image): The input image to be segmented. It can be provided as either a file path or a PIL image object.

        Returns:
            PIL image: The segmented image, where the cat is separated from the background.

        Raises:
            FileNotFoundError: If the image file is not found at the specified path.

        Example usage:
            >>> segmenter = CatSegmenter()
            >>> segmented_image = segmenter.segment("path/to/input_image.jpg")
            >>> segmented_image.show()
        """
        # Load and preprocess the input image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError("Image not found at {}".format(image))
            image = Image.open(image).convert("RGB")
            input_batch = self.preprocess(image).unsqueeze(0).cuda()
            
        elif isinstance(image, Image.Image):
            input_batch = self.preprocess(image).unsqueeze(0).cuda()
            
        else:
            raise TypeError("Image must be a PIL image or a path to an image file, but got {}".format(type(image)))
        
        # Run the image through the model
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        # Post-process the output to obtain the segmentation mask
        output_predictions = output.argmax(0)
        
        # Convert the output predictions to a PIL image
        output_predictions = torch.where(output_predictions == 0, 0, 255)
        mask = transforms.ToPILImage()(output_predictions.byte())
        
        # Display the input image and the segmentation mask
        segment = Image.composite(image, Image.new('RGB', image.size), mask)
        return segment
    
    def display_image(self, image, titles):
        # sanity check
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        # Display the image
        plt.imshow(np.array(image))
        plt.title(titles)
        plt.show()

