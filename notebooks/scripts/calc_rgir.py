# Calculate NDVI in NAIP Images
# Created on Jan 6 2023
# Author: Henry CH, Yeung

import glob, os
import rasterio as rio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Define input image directory
os.chdir(os.getcwd())
baseImgDir = "../../rawImage"
inputImages = glob.glob(os.path.join(baseImgDir, "raw*.tif"))

for im in tqdm(inputImages):
    inputImage = im

    # Load red and NIR bands - note all NAIP 4-band images have band order RGBN
    with rio.open(inputImage) as src:
        band_red = src.read(1)
        band_green = src.read(2)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate RGIr
    rgir = band_red.astype(float) * (band_red.astype(float) / band_green.astype(float))

    # Extract metadata from input image for the output image
    kwargs = src.meta
    kwargs.update(
        dtype='byte',
        count = 1)

    # Define output directory
    outputImgDir = "../../rawImage"
    ouputName_rgir = "rgir_"+os.path.basename(inputImage)

    # Save image
    with rio.open(os.path.join(outputImgDir, ouputName_rgir), 'w', **kwargs) as dst:
            dst.write_band(1, rgir.astype('byte'))