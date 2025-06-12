# Calculate NDVI in NAIP Images, dated 19 May 2023
# Author: Henry CH, Yeung  
# University of Virginia, VA

import glob, os
import rasterio as rio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Define input image directory
os.chdir(os.getcwd())
# baseImgDir = "../../source"
file_prefix = 'NAIP_'
baseImgDir = '/scratch/apm5rt/dryland_U-Net_NorthCarolinaDeadOnly2016_SM/naip_all/naipImage_SC_2026'
inputImages = glob.glob(os.path.join(baseImgDir, file_prefix+"*.tif"))

for im in tqdm(inputImages):
    inputImage = im

    outputImgDir = baseImgDir
    outputName_ndvi = inputImage.replace(file_prefix, "NDVI_")
    if not os.path.isfile(outputName_ndvi):
        # Load red and NIR bands - note all NAIP 4-band images have band order RGBN
        with rio.open(inputImage) as src:
            band_red = src.read(1)
            band_nir = src.read(4)

        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate NDVI
        ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir.astype(float) + band_red.astype(float))
        # ndvi *= 1000
        # ndvi = np.floor(ndvi)

        ndvi = np.floor((ndvi + 1) * 128)  # [-1 1] -> [0 256]
        ndvi[ndvi < 0] = 0;             # not really necessary, just in case & for symmetry
        ndvi[ndvi > 255] = 255;         # in case the original value was exactly 1

        # Extract metadata from input image for the output image
        kwargs = src.meta
        kwargs.update(
            dtype=rio.uint8,
            count = 1,
            nodata=0)
        
        # Save image
        with rio.open(os.path.join(outputImgDir, outputName_ndvi), 'w', **kwargs) as dst:
                dst.write_band(1, ndvi.astype(rio.uint8))

    else:
        print(os.path.basename(outputName_ndvi)+": File already processed!")
