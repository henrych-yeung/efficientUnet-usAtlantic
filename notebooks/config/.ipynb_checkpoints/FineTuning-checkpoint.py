import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        
        # For loading pretrianed network
        self.trained_model_dir = './saved_models/UNet_FINAL/'
        self.trained_model_name = 'trees_efficientnetb5pt_20231105-1716_Adam_weightmap_ftversky3708_12345_256'
        self.model_file_type = '.h5'
        self.PREPROCESS = None
        
        # For reading the ndvi, pan and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        self.base_dir_train_BASE = '/scratch/apm5rt/dryland_U-Net_NorthCarolinaDeadOnly2016_SM/output_train_normalize'
        self.base_dir_val_BASE = '/scratch/apm5rt/dryland_U-Net_NorthCarolinaDeadOnly2016_SM/output_val_normalize'
        self.base_dir_train_FT = '/scratch/apm5rt/dryland_U-Net_NorthCarolinaDeadOnly2016_SM/output_train_FineTuning'
        self.base_dir_val_FT = '/scratch/apm5rt/dryland_U-Net_NorthCarolinaDeadOnly2016_SM/output_val_FineTuning'
        self.image_type = '.png'
        self.ndvi_fn = 'ndvi'
        self.r_fn = 'r'
        self.g_fn = 'g'
        self.b_fn = 'b'
        self.n_fn = 'nir'
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.seed = 1
        self.subsample_frac = 1
        self.start_trainable = 400
        
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        self.patch_size = (256,256,7) # Height * Width * (Input + Output) channels
        # # When stratergy == sequential, then you need the step_size as well
        # step_size = (128,128)
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio. 
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 1

        
        # The split of training areas into training, validation and testing set, is cached in patch_dir.
        self.patch_dir = './patches{}'.format(self.patch_size[0])
        self.frames_json = os.path.join(self.patch_dir,'frames_list_nosplit.json')
        
        # Shape of the input data, height*width*channel; Here channels are NVDI and Pan
        self.input_shape = (256,256,4)
        self.input_image_channel = [1,2,3,4]
        self.input_label_channel = [5]
        self.input_weight_channel = [6]
        
        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 12
        self.NB_EPOCHS = 500

        # number of validation images to use
        self.VALID_IMG_COUNT = 500
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 1000

