# Configuration of the parameters for the 3-FinalRasterAnalysis.ipynb notebook
class Configuration:
    '''
    Configuration for the notebook where objects are predicted in the image.
    Copy the configTemplate folder and define the paths to input and output data.
    '''
    def __init__(self):

        # Input related variables
        self.input_image_dir = '/scratch/apm5rt/NAIP_1m'
        #self.input_image_dir = '/project/SDS/capstones_yang/us-tree-detection/data/training/organized_1m/images' #For evaluation
        self.input_image_type = '.tif'
        self.ndvi_fn_st = 'm_'
        self.pan_fn_st = 'm_'
        #self.ndvi_fn_st = 'output_eval_' #For evaluation
        #self.pan_fn_st = 'output_eval_' #For evaluation
        self.trained_model_dir = '/scratch/apm5rt/efficientUnet-usAtlantic/notebooks/saved_models'
        self.trained_model_name = 'trees_efficientnetb5pt_20231105-1716_Adam_weightmap_ftversky3708_12345_256_fine_20240322-1736_0_1'
        self.model_file_type = '.h5'
        self.PREPROCESS = None

        # Output related variables
        self.output_dir = "/scratch/apm5rt/segmentation_1m"
        #self.output_dir = '/project/SDS/capstones_yang/us-tree-detection/data/testing/crownSegment_1m/all_test_image_split_ws2d5' #For evaluation
        self.output_image_type = '.tif'
        self.output_prefix = 'poly_m_'
        self.output_shapefile_type = '.gpkg'
        self.overwrite_analysed_files = False
        self.output_dtype='uint8'
        self.split_crowns=True

        # Variables related to batches and model
        self.TTA = False # Perform Test-time Augmentation (Processing time would be longer if True)
        self.OPERATOR = 'MAX' # Get maximum or minimum prediction probability
        self.BATCH_SIZE = 32 # Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=256 # Should be same as the WIDTH used for training the model
        self.HEIGHT=256 # Should be same as the HEIGHT used for training the model
        self.STRIDE=196 #224 or 196   # STRIDE = WIDTH means no overlap, STRIDE = WIDTH/2 means 50 % overlap in prediction
