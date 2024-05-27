import cv2
import numpy as np
import os
import pandas as pd

def compute_HOG(image_path):
    
    image = cv2.imread(image_path)
    
    # Step 1: Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Calculate gradient magnitude and orientation
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Step 3: Define parameters for HOG
    cell_size = (8, 8)  # Size of cell in pixels
    block_size = (2, 2)  # Size of block in cells
    nbins = 9  # Number of orientation bins
    
    # Step 4 & 5: Compute HOG descriptors
    hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1] * cell_size[1],
                                      gray.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    hog_features = hog.compute(gray)
    
    return hog_features