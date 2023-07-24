#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Gaussian_Histogram():
    def __init__(self):
        pass
    
    def perform_gauss(self, image, label):
        if isinstance(image, str):
            image = Image.open(image).convert("L")
            image_array = np.array(image)
        elif isinstance(image, Image.Image):
            image_array = np.array(image)

        hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])
        
        mu = np.mean(image_array)
        sigma = np.std(image_array)
        # Generate Gaussian filter
        gaussian_filter = self.gaussian(bins, mu, sigma)
        # Plot the histogram and the Gaussian filter
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
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))