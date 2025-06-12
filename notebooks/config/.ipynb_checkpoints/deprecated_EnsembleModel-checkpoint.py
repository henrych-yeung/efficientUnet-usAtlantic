import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the ndvi, pan and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        self.base_dir_train = '../output_train'
        self.base_dir_val = '../output_val'
        self.image_type = '.png'
        self.ndvi_fn = 'ndvi'
        self.r_fn = 'r'
        self.g_fn = 'g'
        self.b_fn = 'b'
        self.n_fn = 'nir'
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        self.patch_size = (256,256,7) # Height * Width * (Input + Output) channels
        # When stratergy == sequential, then you need the step_size as well
        self.step_size = (256,256)
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio. 
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 0.3

        self.model_path = './saved_models/UNet/'
        self.model1_name = 'trees_resnet101pt_20230506-1836_Adam_weightmap_ftversky5575_012345_256'
        self.model2_name = 'trees_inceptionv3pt_20230506-1222_Adam_weightmap_ftversky5575_012345_256'
        self.model3_name = 'trees_efficientnetb3pt_20230506-0025_Adam_weightmap_ftversky5575_012345_256'
        self.model_type = '.h5'
        
#         # The split of training areas into training, validation and testing set, is cached in patch_dir.
#         self.patch_dir = './patches{}'.format(self.patch_size[0])
#         self.frames_json = os.path.join(self.patch_dir,'frames_list_nosplit.json')

        #Backbone of CNN encoder (Also defines the decoder architecture)
        self.BACKBONE = 'resnet101'
        self.PRETRAIN = 'imagenet'
        self.PREPROCESS = None

        # Shape of the input data, height*width*channel; Here channels are NVDI and Pan
        self.input_shape = (256,256,5)
        self.input_image_channel = [0,1,2,3,4]
        self.input_label_channel = [5]
        self.input_weight_channel = [6]
        
        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
#         self.BATCH_SIZE = 16
        self.NB_EPOCHS = 200

        # number of validation images to use
        self.VALID_IMG_COUNT = 350
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 900

