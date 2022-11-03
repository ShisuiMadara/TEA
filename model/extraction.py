import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

threshold = 0.6

def filter_boxes(boxes, box_confidence, box_class_probs, threshold):

    box_scores = box_confidence * box_class_probs
    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    
 
    filtering_mask = box_class_scores > threshold
    
 
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes



def iou(box1, box2):

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

   
    xi1 = np.maximum(box1_x1, box2_x1)
    yi1 = np.maximum(box1_y1, box2_y1)
    xi2 = np.minimum(box1_x2, box2_x2)
    yi2 = np.minimum(box1_y2, box2_y2)
    inter_width = np.maximum(xi2 - xi1, 0)
    inter_height =  np.maximum(yi2-yi1, 0)
    inter_area = inter_width*inter_height
    

    box1_area = np.abs(box1_x2 - box1_x1) * np.abs(box1_y1 - box1_y2)
    box2_area = np.abs(box2_x1 - box2_x2) * np.abs(box2_y1 - box2_y2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area/union_area
    
    return iou



def non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')    

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=max_boxes, iou_threshold=iou_threshold)
   
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes

