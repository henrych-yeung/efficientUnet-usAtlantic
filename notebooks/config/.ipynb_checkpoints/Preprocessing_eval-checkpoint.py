
import os


# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons
        self.training_base_dir = '/Volumes/Extreme Pro/ghostForest_usAtlantic/code_atlanticGhostDL/training_all/ver_20240313'
        self.training_area_fn = 'trainingArea_eval_buffer_FINAL2.gpkg'
        self.training_polygon_fn = 'trainingPolygon_eval_FINAL2.gpkg'

        # For reading the VHR images
        self.bands = [0,1,2,3]
        self.raw_image_base_dir = '/Volumes/Extreme Pro/ghostForest_usAtlantic/code_atlanticGhostDL/trainingImage_all'
        self.raw_image_file_type = '.tif'
        self.raw_ndvi_image_prefix = 'reproj_ndvi'
        self.raw_pan_image_prefix = 'reproj_raw'

        # For writing the extracted images and their corresponding annotations and boundary file
        self.path_to_write = '/Volumes/Extreme Pro/ghostForest_usAtlantic/code_atlanticGhostDL/output_eval'
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
