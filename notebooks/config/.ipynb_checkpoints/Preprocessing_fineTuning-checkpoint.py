
import os


# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons from BASE dataset
        self.training_base_dir = '../training/FINAL'
        self.training_area_fn = 'trainingArea_train_FINAL.shp'
        self.training_polygon_fn = 'trainingPolygon_FINAL.shp'

        # For reading the training areas and polygons from FINE-TUNING dataset
        self.training_base_FT_dir = '../training/FINE_TUNING'
        self.training_area_FT_fn = 'trainingArea_train_FineTuning.shp'
        self.training_polygon_FT_fn = 'trainingPolygon_FineTuning.shp'

        # For subsampling BASE datset
        self.seed_value = 1
        self.subsample_frac = 0.3 #fraction of the dataset you want to sample (e.g., 0.5 for 50%)

        # For reading the VHR images
        self.bands = [0,1,2,3]
        self.raw_image_base_dir = '../trainingImage_FINETUNING'
        self.raw_image_file_type = '.tif'
        self.raw_ndvi_image_prefix = 'reproj_ndvi'
        self.raw_pan_image_prefix = 'reproj_raw'

        # For writing the extracted images and their corresponding annotations and boundary file
        self.path_to_write = '../output_train_FineTuning'
        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.png'
        self.extracted_ndvi_filename = 'ndvi'
        self.extracted_pan_filename = ['r', 'g', 'b', 'nir']
        self.extracted_annotation_filename = 'annotation'
        self.extracted_boundary_filename = 'boundary'
        

        # Path to write should be a valid directory
        if not os.path.exists(self.path_to_write):
            os.makedirs(self.path_to_write)
        assert os.path.exists(self.path_to_write)

        if not len(os.listdir(self.path_to_write)) == 0:
            print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')
