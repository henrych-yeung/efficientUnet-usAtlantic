# Configuration of the parameters for the 3-FinalRasterAnalysis.ipynb notebook
class Configuration:
    '''
    Configuration for the notebook where objects are predicted in the image.
    Copy the configTemplate folder and define the paths to input and output data.
    '''
    def __init__(self):
        
        # Input related variables
        self.input_image_dir = '../rawImage2020_reproj'
        self.input_image_type = '.tif'
        self.ndvi_fn_st = 'repro_ndvi_'
        self.pan_fn_st = 'repro_raw_'
        self.trained_model_dir = './saved_models/UNet_normalize/'
        self.trained_model1_name = 'trees_resnet101pt_20230520-2122_Adam_weightmap_ftversky5575_012345_256'
        self.trained_model2_name = 'trees_inceptionv3pt_20230520-2221_Adam_weightmap_ftversky5575_012345_256'
        self.trained_model3_name = 'trees_efficientnetb3pt_20230520-1830_Adam_weightmap_ftversky5575_012345_256'
        self.model_file_type = '.h5'
        self.model_weights = [0.333, 0.333, 0.334]
        self.PREPROCESS = None
        self.THRESHOLD = 0.7

        # Output related variables
        self.output_dir = '../detection2020'
        self.output_image_type = '.tif'
        self.output_prefix = 'det_ens_th07_'
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files = False
        self.output_dtype='uint8'

        # Variables related to batches and model
        self.BATCH_SIZE = 8 # Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=256 # Should be same as the WIDTH used for training the model
        self.HEIGHT=256 # Should be same as the HEIGHT used for training the model
        self.STRIDE=224 #224 or 196   # STRIDE = WIDTH means no overlap, STRIDE = WIDTH/2 means 50 % overlap in prediction
