# Calculate Green Red Difference Index in NAIP Images
# Created on March 2 2023
# Author: Henry CH, Yeung

import glob, os
import rasterio as rio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Define input image directory
os.chdir(os.getcwd())
baseImgDir = "../../source"
inputImages = glob.glob(os.path.join(baseImgDir, "*.tif"))

for im in tqdm(inputImages):
    inputImage = im

    # Load red and NIR bands - note all NAIP 4-band images have band order RGBN
    with rio.open(inputImage) as src:
        band_red = src.read(1)
        band_green = src.read(2)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate GRDI
    rgdi = (band_green.astype(float) - band_red.astype(float)) / (band_green.astype(float) + band_red.astype(float))

    # Extract metadata from input image for the output image
    kwargs = src.meta
    kwargs.update(
        dtype=rio.float32,
        count = 1)

    # Define output directory
    outputImgDir = "../../rawImage"
    ouputName = "grdi_"+os.path.basename(inputImage)

    # Save image
    with rio.open(os.path.join(outputImgDir, ouputName), 'w', **kwargs) as dst:
            dst.write_band(1, rgdi.astype(rio.float32))