#!/usr/bin/env python
# coding: utf-8

#    Edited by Henry CH Yeung
#    
#    Author: Ankit Kariryaa, University of Bremen

# In[1]:
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

import rasterio
from rasterio import windows
from rasterio.features import shapes
from rasterio.warp import transform as warp_transform

import geopandas as gps
import numpy as np
from tqdm import tqdm
from itertools import product
from tensorflow.keras.models import load_model
from shapely.geometry import Point, Polygon, mapping, shape
import matplotlib.pyplot as plt
import warnings, logging, argparse

from core.UNet import UNet
from core.losses import focal_tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info import FrameInfo, image_normalize, image_standardize
from core.dataset_generator import DataGenerator
from core.split_frames import split_dataset
from core.visualize import display_images
from segmentation_models import get_preprocessing

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
print(tf.__version__)


# In[2]:


# Required configurations (including the input and output paths) are stored in a separate file (such as config/RasterAnalysis.py)
# Please provide required info in the file before continuing with this notebook. 
 
from config import RasterAnalysis
# In case you are using a different folder name such as configLargeCluster, then you should import from the respective folder 
# Eg. from configLargeCluster import RasterAnalysis

config = RasterAnalysis.Configuration()


# In[3]:


# Load a pretrained model
OPTIMIZER = adam
trained_model_path = os.path.join(config.trained_model_dir, config.trained_model_name+config.model_file_type)
model = load_model(trained_model_path, custom_objects={'ftversky': focal_tversky}, compile=False)
# model.compile(optimizer=OPTIMIZER, loss=tversky)
print(trained_model_path)
model.summary()


# In[4]:


# Methods to add results of a patch to the total results of a larger area. The operator could be min (useful if there are too many false positives), max (useful for tackle false negatives)
def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    # currValue[:] = -1 #Comment this line in case of MAX
    newPredictions = prediction[:he, :wi]
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == 0] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions) 
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    else: #operator == 'REPLACE':
        resultant = newPredictions    
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get. 
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he, col:col+wi] =  resultant
    
    # plt.subplot(1, 3, 1)
    # plt.imshow(currValue, cmap='viridis', vmin=0, vmax=1)
    # plt.title('Current Value Before Update')
    # plt.subplot(1, 3, 2)
    # plt.imshow(newPredictions, cmap='viridis', vmin=0, vmax=1)
    # plt.title('newPredictions')
    # plt.subplot(1, 3, 3)
    # plt.imshow(resultant, cmap='viridis', vmin=0, vmax=1)
    # plt.title('Result')
    # plt.show()
    return (res)


# In[5]:


# Methods that actually makes the predictions
def predict_using_model(model, batch, batch_pos, mask, operator, tta=False):
    tm = np.stack(batch, axis = 0)
    if tta:
        prediction_ori = model.predict(tm)
        
        # Horizontal flip
        prediction_lr = model.predict(np.flip(tm, axis=2))
        prediction_lr = np.flip(prediction_lr, axis=2)
        
        # # Contrast
        # prediction_ct = model.predict(tm * 0.85)

        # Vertical flip
        prediction_ud = model.predict(np.flip(tm, axis=1))
        prediction_ud = np.flip(prediction_ud, axis=1)

        # Vertical+Horizontal flip
        prediction_lrud = model.predict(np.flip(np.flip(tm, axis=1), axis=2))
        prediction_lrud = np.flip(np.flip(prediction_lrud, axis=1), axis=2)
        
#         # Rotation (+/- 90°)
#         prediction_r0 = model.predict(np.rot90(tm, k=1, axes=(2,1)))
#         prediction_r0 = np.rot90(prediction_r0, k=-1, axes=(2,1))
# #         prediction_r1 = model.predict(np.rot90(tm, k=-1, axes=(2,1)))
# #         prediction_r1 = np.rot90(prediction_r1, k=1, axes=(2,1))

        # Average result
        prediction = (prediction_ori + prediction_lr + prediction_ud + prediction_lrud) / 4
        
    else:
        prediction = model.predict(tm)
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask

def detect_tree(ndvi_img, pan_img, width=256, height=256, stride = 128, normalize=True, preprocess=None):
    assert ndvi_img.meta['width'] == pan_img.meta['width'] and ndvi_img.meta['height'] == pan_img.meta['height']
    nols, nrows = ndvi_img.meta['width'], ndvi_img.meta['height']
    meta = ndvi_img.meta.copy()
    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
        meta['dtype'] = np.float32
    if preprocess: preprocess_input = get_preprocessing(preprocess)
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     print(nrows, nols)

    mask = np.zeros((nrows, nols), dtype=meta['dtype'])
#     mask = mask -1 # Note: The initial mask is initialized with -1 instead of zero to handle the MIN case (see addToResult)
    batch = []
    batch_pos = [ ]
    for col_off, row_off in  tqdm(offsets):
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        if window.width < 2:
            continue
        else:
            transform = windows.transform(window, ndvi_img.transform)
            patch = np.zeros((height, width, 4)) #Add zero padding in case of corner images
            ndvi_sm = ndvi_img.read(window=window)[[0],...]
            pan_sm = pan_img.read(window=window)
    #         plt.imshow(np.squeeze(ndvi_sm))
    #         plt.show()
            r_sm, g_sm, b_sm, n_sm = pan_sm[[0],...], pan_sm[[1],...], pan_sm[[2],...], pan_sm[[3],...]
            temp_im = np.stack((r_sm, g_sm, b_sm, n_sm), axis = -1)
            temp_im = np.squeeze(temp_im)

            if normalize:
    #             temp_im = image_standardize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image to [0,1] independently per channel
            if preprocess: temp_im = preprocess_input((255*temp_im).astype(int))

            patch[:window.height, :window.width] = temp_im
            batch.append(patch)
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
            if (len(batch) == config.BATCH_SIZE):
                mask = predict_using_model(model, batch, batch_pos, mask, config.OPERATOR, config.TTA)
                batch = []
                batch_pos = []
    
    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model(model, batch, batch_pos, mask, config.OPERATOR, config.TTA)
        batch = []
        batch_pos = []
    return(mask, meta)


# In[6]:


schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'str', 'canopy': 'float:15.2',},
    }

def drawPolygons(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=255, fill=255)
    mask = np.array(mask)#, dtype=bool)   
    return(mask)

def transformToXY(polygons, transform):
    tp = []
    for polygon in polygons:
        rows, cols = zip(*polygon)
        x,y = rasterio.transform.xy(transform, rows, cols)
        tp.append(list(zip(x,y)))
    return (tp)

def createShapefileObject(polygons, meta, wfile):
    with fiona.open(wfile, 'w', crs=meta.get('crs').to_dict(), driver='ESRI Shapefile', schema=schema) as sink:
        for idx, mp in enumerate(polygons):
            try:
#                 poly = Polygon(poly)
    #             assert mp.is_valid
    #             assert mp.geom_type == 'Polygon'
                sink.write({
                    'geometry': mapping(mp),
                    'properties': {'id': str(idx), 'canopy': mp.area},
                })
            except:
                print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
#                 print(mp)

# Generate a mask with polygons
def transformContoursToXY(contours, transform = None):
    tp = []
    for cnt in contours:
        pl = cnt[:, 0, :]
        cols, rows = zip(*pl)
        x,y = rasterio.transform.xy(transform, rows, cols)
        tl = [list(i) for i in zip(x, y)]
        tp.append(tl)
    return (tp)


def mask_to_polygons(maskF, transform):
    # first, find contours with cv2: it's much faster than shapely
    th = 0.5
    mask = maskF.copy()
    mask[mask < th] = 0
    mask[mask >= th] = 1
    mask = ((mask) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    #Convert contours from image coordinate to xy coordinate
    contours = transformContoursToXY(contours, transform)
    if not contours: #TODO: Raise an error maybe
        print('Warning: No contours/polygons detected!!')
        return [Polygon()]
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []

    for idx, cnt in enumerate(contours):
        if idx not in child_contours: #and cv2.contourArea(cnt) >= min_area: #Do we need to check for min_area??
            try:
                poly = Polygon(
                    shell=cnt,
                    holes=[c for c in cnt_children.get(idx, [])])
                           #if cv2.contourArea(c) >= min_area]) #Do we need to check for min_area??
                all_polygons.append(poly)
            except:
                pass
#                 print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
    print(len(all_polygons))
    return(all_polygons)

def create_contours_shapefile(mask, meta, out_fn):
    res = mask_to_polygons(mask, meta['transform'])
#     res = transformToXY(contours, meta['transform'])
    createShapefileObject(res, meta, out_fn)


# def writeMaskToDisk(detected_mask, detected_meta, wp, write_as_type = 'uint8', th = 0.5, create_countors = False):
#     # Convert to correct required before writing
#     if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
#         print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
#         detected_mask[detected_mask<th]=0
#         detected_mask[detected_mask>=th]=1
#         detected_mask = detected_mask.astype(write_as_type)
#         detected_meta['dtype'] =  write_as_type
#         detected_meta['count'] = 1
    # with rasterio.open(wp, 'w', **detected_meta) as outds:
    #     outds.write(detected_mask, 1)
    # if create_countors:
    #     wp = wp.replace(image_type, output_shapefile_type)
    #     create_contours_shapefile(detected_mask, detected_meta, wp)
    
from skimage.morphology import remove_small_holes
from shapely.geometry import shape, Polygon
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import scipy.ndimage as ndi

def writeMaskToDisk(mask, meta, out_path, threshold=0.5, min_area=0.01, hole_size_m2=100, win_size_m=2.5, split_crowns=True):
    """
    Convert prediction mask to polygon GeoPackage with optional crown splitting.

    Args:
        mask (np.ndarray): 2D prediction mask.
        meta (dict): Raster metadata.
        out_path (str): Output path (.gpkg).
        threshold (float): Threshold for binarization.
        min_area (float): Minimum area to retain.
        hole_size (int): Max hole size to fill (in pixels).
        split_crowns (bool): Whether to apply watershed-based crown separation.
    """
    
    # Get pixel size (assume square pixels)
    res = abs(meta["transform"][0])  # meters/pixel
    
    # Convert m² to pixel area
    hole_size_px = int(hole_size_m2 / (res * res))
    
    # Convert 10 m to pixel window size
    win_size_px = max(3, int(win_size_m / res))  # must be odd and ≥ 3
    
    
    print(f"Saving GPKG to {out_path}")
    binary = (mask >= threshold)
    binary = remove_small_holes(binary, area_threshold=hole_size_px, connectivity=1)
    binary = binary.astype(np.uint8)

    if split_crowns:
        print("⚡ Splitting merged crowns with watershed...")
        dist_transform = ndi.distance_transform_edt(binary)
        
        coords = peak_local_max(dist_transform, 
        #    footprint=np.ones((win_size_px, win_size_px)), 
            min_distance= win_size_px,
            labels=binary)
        peak_mask = np.zeros_like(dist_transform, dtype=bool)
        peak_mask[tuple(coords.T)] = True

        # Use 8-connectivity for marker labeling
        markers, num_markers = ndi.label(peak_mask, structure=np.ones((3, 3)))

        # Use same connectivity in watershed
        binary_split = watershed(-dist_transform, markers=markers, mask=binary,
            connectivity=2,  # optional: use 2 for full 8-connectivity
            watershed_line=False
        )

    binary = np.maximum(binary, binary_split)
    
    transform = meta["transform"]
    crs = meta["crs"]
    shape_gen = shapes(binary, mask=(binary > 0), transform=transform)

    polygons = []
    for geom, val in shape_gen:
        if val == 0:
            continue
        poly = shape(geom)
        if poly.area >= min_area:
            poly = Polygon(poly.exterior)
            polygons.append(poly)

    if not polygons:
        print("⚠️ No valid polygons found — writing empty file.")
        schema = {"geometry": "Polygon", "properties": {"id": "int", "area": "int"}}
        gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
        gdf.to_file(out_path, schema=schema, driver="GPKG")
    else:
        gdf = gpd.GeoDataFrame({
            "id": range(1, len(polygons)+1),
            "area": [p.area for p in polygons]
        }, geometry=polygons, crs=crs)
        gdf.to_file(out_path, driver="GPKG")
    
    print(f"✅ Saved {len(gdf)} polygons to {out_path}")

    
# In[7]:


# # Predict trees in the all the files in the input image dir
# # Depending upon the available RAM, images may not to be split before running this cell.
# # Use the Auxiliary-2-SplitRasterToAnalyse if the images are too big to be analysed in memory.

# import time
# from multiprocessing import Pool

# all_files = []
# for root, dirs, files in os.walk(config.input_image_dir):
# #     print(files)
#     for file in files:
#         if file.endswith(config.input_image_type) and file.startswith(config.ndvi_fn_st):
#              all_files.append((os.path.join(root, file), file))

# zipped = list(zip(all_files))

# def detect_in_parallel(all_files_zip):
#     for (fullPath, filename), in all_files_zip:
#         outputFile = os.path.join(config.output_dir, filename.replace(config.ndvi_fn_st, config.output_prefix))
#         if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
#             with rasterio.open(fullPath) as ndvi:
#                 with rasterio.open(fullPath.replace(config.ndvi_fn_st, config.pan_fn_st)) as pan:
#                     print(fullPath)
#                     detectedMask, detectedMeta = detect_tree(ndvi, pan, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, preprocess=config.PREPROCESS) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
#     #               #Write the mask to file
#                     writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
#         else:
#             print('File already analysed!', fullPath)


            
# def main():
#     ncpus=int(10)
#     print ('ncpus={}'.format(ncpus))

#     pool = Pool(processes=ncpus)
#     tic=time.time()
#     results = pool.starmap(detect_in_parallel, zipped)
#     toc=time.time()
#     pool.close(); pool.join()
#     print("Parallel time on "+str(ncpus)+" cores:"+str(round(toc-tic,4)))
# if __name__=="__main__":
#     main()


# In[8]:


# # Predict trees in the all the files in the input image dir
# # Depending upon the available RAM, images may not to be split before running this cell.
# # Use the Auxiliary-2-SplitRasterToAnalyse if the images are too big to be analysed in memory.

# import time
# from multiprocessing import Pool

# all_files = []
# for root, dirs, files in os.walk(config.input_image_dir):
# #     print(files)
#     for file in files:
#         if file.endswith(config.input_image_type) and file.startswith(config.ndvi_fn_st):
#              all_files.append((os.path.join(root, file), file))

# def detect_in_parallel(fullPath, filename):
#     outputFile = os.path.join(config.output_dir, filename.replace(config.ndvi_fn_st, config.output_prefix))
#     if not os.path.isfile(outputFile) or config.overwrite_analysed_files: 
#         with rasterio.open(fullPath) as ndvi:
#             with rasterio.open(fullPath.replace(config.ndvi_fn_st, config.pan_fn_st)) as pan:
#                 print(fullPath)
#                 detectedMask, detectedMeta = detect_tree(ndvi, pan, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, preprocess=config.PREPROCESS) # WIDTH and HEIGHT should be the same and in this case Stride is 50 % width
# #               #Write the mask to file
#                 writeMaskToDisk(detectedMask, detectedMeta, outputFile, write_as_type = config.output_dtype, th = 0.5, create_countors = False)                
#     else:
#         print('File already analysed!', fullPath)


         
# def main():
#     ncpus=int(10)
#     print ('ncpus={}'.format(ncpus))

#     pool = Pool(processes=ncpus)
#     tic=time.time()
#     results = pool.starmap(detect_in_parallel, [(fullPath, filename) for fullPath, filename in all_files])
#     toc=time.time()
#     pool.close(); pool.join()
#     print("Parallel time on "+str(ncpus)+" cores:"+str(round(toc-tic,4)))
# if __name__=="__main__":
#     main()


# In[9]:


# Predict trees in the all the files in the input image dir
# Depending upon the available RAM, images may not to be split before running this cell.
# Use the Auxiliary-2-SplitRasterToAnalyse if the images are too big to be analysed in memory.

####### Original Version (No parallel processing) #######
import time, random

def main():
    print('🚀 Starting raster analysis...')

    # --- Load default config ---
    config = RasterAnalysis.Configuration()

    # --- Parse CLI arguments (optional override) ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input directory containing image tiles")
    parser.add_argument("--output", help="Output directory for predicted .gpkg files")
    parser.add_argument("--stable_min", type=float, default=0,
                    help="Minimum file age (minutes) before processing (to skip files still being written)")
    args = parser.parse_args()

    # --- Use CLI arguments if provided, else fallback to config defaults ---
    input_dir = args.input or config.input_image_dir
    output_dir = args.output or config.output_dir

    print(f"📂 Input directory: {input_dir}")
    print(f"💾 Output directory: {output_dir}")

    # --- Update config paths dynamically ---
    config.input_image_dir = input_dir
    config.output_dir = output_dir

    # --- Ensure output directory exists ---
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Collect all NDVI files ---
    all_files = []
    for root, _, files in os.walk(config.input_image_dir):
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.ndvi_fn_st):
                all_files.append((os.path.join(root, file), file))

    if not all_files:
        print(f"⚠️ No input files found in {config.input_image_dir}")
        return

    # --- Filter out files modified within the last N minutes ---
    if args.stable_min > 0:
        now = time.time()
        stable_files = []
        for fullPath, filename in all_files:
            try:
                age_min = (now - os.path.getmtime(fullPath)) / 60
                if age_min >= args.stable_min:
                    stable_files.append((fullPath, filename))
                else:
                    print(f"⏸️ Skipping (modified {age_min:.1f} min ago): {filename}")
            except Exception as e:
                print(f"⚠️ Could not check modification time for {filename}: {e}")

        all_files = stable_files
        print(f"✅ Found {len(all_files)} TIFFs older than {args.stable_min} min.")
        if not all_files:
            print("⚠️ No stable TIFFs found. Exiting.")
            return


    # --- Shuffle order: each server gets different random sequence ---
    random.seed(os.getpid() + int(time.time()))
    random.shuffle(all_files)
    
    # --- Process each file ---
    for fullPath, filename in all_files:
        rel_dir = os.path.relpath(os.path.dirname(fullPath), config.input_image_dir)

        # Add "_vect" directly to the last folder in the relative path
        parent_dir = os.path.dirname(rel_dir)
        folder_name = os.path.basename(rel_dir)
        rel_dir_vect = os.path.join(parent_dir, folder_name + "_vect") if parent_dir else folder_name + "_vect"

        out_subdir = os.path.join(config.output_dir, rel_dir_vect)
        os.makedirs(out_subdir, exist_ok=True)

        outputFile = os.path.join(out_subdir, filename.replace(config.ndvi_fn_st, config.output_prefix))
        gpkg_path = outputFile.replace(config.output_image_type, config.output_shapefile_type)

        # Skip if output already exists and overwrite is disabled
        if os.path.isfile(gpkg_path) and not config.overwrite_analysed_files:
            print(f"⏩ Skipping already processed file: {gpkg_path}")
            continue

        try:
            with rasterio.open(fullPath) as ndvi, rasterio.open(fullPath.replace(config.ndvi_fn_st, config.pan_fn_st)) as pan:
                print(f"🛰️ Processing: {fullPath}")
                detectedMask, detectedMeta = detect_tree(
                    ndvi,
                    pan,
                    width=config.WIDTH,
                    height=config.HEIGHT,
                    stride=config.STRIDE,
                    preprocess=config.PREPROCESS
                )

                # Overwrite if file appears during processing
                if os.path.isfile(gpkg_path):
                    print(f"⚠️ Output already exists, overwriting: {gpkg_path}")

                writeMaskToDisk(
                    detectedMask,
                    detectedMeta,
                    gpkg_path,
                    threshold=0.5,
                    split_crowns=config.split_crowns
                )

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("✅ All processing complete!")


if __name__ == "__main__":
    main()
    print('endddddd')


