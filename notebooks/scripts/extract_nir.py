# Extract NIR band in NAIP Images
# Created on Jan 6 2023
# Author: Henry CH, Yeung

import glob, os
import rasterio as rio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Define input image directory
os.chdir(os.getcwd())
# baseImgDir = "../../source"
baseImgDir = "/Users/apm5rt/OneDrive - University of Virginia/data/trial_NC/NAIP/NC2016"
inputImages = glob.glob(os.path.join(baseImgDir, "*.tif"))

for im in tqdm(inputImages):
    inputImage = im

    # Load NIR band note all NAIP 4-band images have band order RGBN
    with rio.open(inputImage) as src:
        band_nir = src.read(4)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Extract metadata from input image for the output image
    kwargs = src.meta
    kwargs.update(
        count = 1)

    # Define output directory
    # outputImgDir = "../../rawImage"
    outputImgDir = "/Users/apm5rt/Library/CloudStorage/OneDrive-UniversityofVirginia/code_deepLearning/dryland_U-Net_NorthCarolinaDeadOnly/rawImage"
    ouputName_ndvi = "nir_"+os.path.basename(inputImage)

    # Save image
    with rio.open(os.path.join(outputImgDir, ouputName_ndvi), 'w', **kwargs) as dst:
            dst.write_band(1, band_nir)