#    Edited by Henry CH Yeung
#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen


import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

def focal_tversky(y_true, y_pred, alpha=0.3, beta=0.7, gamma=0.8, penalty=7):
    """
    Function to calculate the Focal Tversky Loss for imbalanced data in multiclass segmentation
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param gamma: focus parameter for down-weighting easy samples (focal strength). gamma < 1 focuses on more difficult samples. gamma > 1 pays less focus on difficult samples.
    :param weight_map:
    :return: the loss
    """
    
    y_t = y_true[...,0]
    y_t = y_t[..., np.newaxis]
    y_weights = y_true[...,1] * penalty
    y_weights = y_weights[..., np.newaxis]

    ones = 1 
    p0 = y_pred # probability that voxels are class i
    p1 = ones - p0  # probability that voxels are not class i
    g0 = y_t # 1 for a class i voxel
    g1 = ones - g0 # 0 for a non-class i voxel
    
    tp = tf.reduce_sum(y_weights * p0 * g0, axis=(0,1,2)) # no. of true positive pixels
    fp = tf.reduce_sum(y_weights * p0 * g1, axis=(0,1,2)) # no. of false positive pixels
    fn = tf.reduce_sum(y_weights * p1 * g0, axis=(0,1,2)) # no. of false negative pixels
    
    # Tversky Similarity Index (tsi)
    EPSILON = 0.00001
    numerator = tp
    denominator = tp + alpha * fp + beta * fn + EPSILON
    tsi = numerator / denominator

    return K.mean(K.pow((1 - tsi), gamma))

def tversky(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    # weights
    y_weights = y_true[...,1]
    y_weights = y_weights[...,np.newaxis]
    
    ones = 1 
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t
    
    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)
    
    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator

    return 1.0 - tf.reduce_mean(score)

def accuracy(y_true, y_pred):
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))

def dice_coef(y_true, y_pred, smooth=0.0000001):
    """compute dice coef"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)

def dice_loss(y_true, y_pred):
    """compute dice loss"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_t, y_pred)

def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((y_t) * (1 - y_pred))

def sensitivity(y_true, y_pred):
    """compute sensitivity, or recall"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def specificity(y_true, y_pred):
    """compute specificity"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))

def precision(y_true, y_pred, smooth=0.0000001):
    """compute precision"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fp) + smooth)

def recall(y_true, y_pred, smooth=0.0000001):
    """compute recall, or sensitivity"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn) + smooth)

def f1_score(y_true, y_pred, smooth=0.0000001):
    """compute recall, or sensitivity"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    p = precision(y_t, y_pred)
    r = recall(y_t, y_pred)
    return (2 * p * r + smooth) / (p + r + smooth)
