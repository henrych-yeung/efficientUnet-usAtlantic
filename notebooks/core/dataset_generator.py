#    Edited by Henry CH Yeung
#    Author: Ankit Kariryaa, University of Bremen

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from segmentation_models import get_preprocessing
import gc

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.4: iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        sometimes(iaa.Crop(percent=(0, 0.3))),  # random crops
        #
        # # Gaussian blur and gamma contrast
        #sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
#         sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),
        
#         sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1), per_channel=True), 0.1),
#         iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
       sometimes(iaa.Multiply((0.8, 1.2), per_channel=True), 0.3),
       sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
#         sometimes(iaa.Add(value=(-0.1,0.1),per_channel=True),0.2),
#         sometimes(iaa.PiecewiseAffine(0.05), 0.3),
#        sometimes(iaa.PerspectiveTransform(0.04), 0.2),
       sometimes(iaa.Affine(
           scale={"x": (0.9, 1.2), "y": (0.9, 1.2)},
           translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
           rotate=(-45, 45)),0.3)
    ],
        random_order=True)
    return seq

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """

    def __init__(self, input_image_channel, patch_size, frame_list, frames, annotation_channel = [2,3], augmenter=None):
        """Datagenerator constructor

        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class.
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined in imageAugmentationWithIAA function.
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter

    # Return all training and label images and weights, generated sequentially with the given step size
    def all_sequential_patches(self, step_size, normalize = 1, BACKBONE=None):
        """Generate all patches from all assigned frames sequentially.

            step_size (tuple(int,int)): Size of the step when generating frames.
            normalize (float): Probability with which a frame is normalized.
        """
        if BACKBONE: preprocess_input = get_preprocessing(BACKBONE)
            
        patches = []
        for fn in self.frame_list:
            frame = self.frames[fn]
            ps = frame.sequential_patches(self.patch_size, step_size, normalize)
            patches.extend(ps)
        data = np.array(patches)
        img = data[..., self.input_image_channel]
        if BACKBONE: img = preprocess_input((255*img).astype(int))
        y = data[..., self.annotation_channel]
        # y would have two channels, i.e. annotations and weights.
        ann = y[...,[0]]
        #boundaries have a weight of 10 other parts of the image has weight 1
        weights = y[...,[1]]
        weights[weights>=0.5] = 10
        weights[weights<0.5] = 1
        ann_joint = np.concatenate((ann,weights), axis=-1)
        return (img, ann_joint)

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE, normalize):
        """Generate patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently).
            normalize (float): Probability with which a frame is normalized.
        """
        patches = []
        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            patch = frame.random_patch(self.patch_size, normalize)
            patches.append(patch)
        data = np.array(patches)

        img = data[..., self.input_image_channel]
        ann_joint = data[..., self.annotation_channel]
        return (img, ann_joint)
#     print("Wrote {} random patches to {} with patch size {}".format(count,write_dir,patch_size))
            
    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def random_generator(self, BATCH_SIZE, normalize=1, BACKBONE=None):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).
            normalize (float): Probability with which a frame is normalized.
        """
        seq = imageAugmentationWithIAA()
        if BACKBONE: preprocess_input = get_preprocessing(BACKBONE)

        while True:
            X, y = self.random_patch(BATCH_SIZE, normalize)
                
            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                # y would have two channels, i.e. annotations and weights. We need to augment y for operations such as crop and transform
                y = seq_det.augment_images(y)
                # Some augmentations can change the value of y, so we re-assign values just to be sure.
                ann =  y[...,[0]]
                ann[ann<0.5] = 0
                ann[ann>=0.5] = 1
                #boundaries have a weight of 10 other parts of the image has weight 1
                weights = y[...,[1]]
                weights[weights>=0.5] = 10
                weights[weights<0.5] = 1
                ann_joint = np.concatenate((ann,weights), axis=-1)
                if BACKBONE: X = preprocess_input((255*X).astype(int))
                yield X, ann_joint
            else:
                # y would have two channels, i.e. annotations and weights.
                ann =  y[...,[0]]
                #boundaries have a weight of 10 other parts of the image has weight 1
                weights = y[...,[1]]
                weights[weights>=0.5] = 10
                weights[weights<0.5] = 1
                ann_joint = np.concatenate((ann,weights), axis=-1)
                if BACKBONE: X = preprocess_input((255*X).astype(int))
                yield X, ann_joint
            gc.collect()




            
            
            
