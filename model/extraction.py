import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import h5py

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

from YAD2K.yad2k.utils.draw_boxes import draw_boxes, get_colors_for_classes
from YAD2K.yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
# from YAD2K.yad2k.utils.utils import scale_boxes, read_classes, read_anchors, preprocess_image

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


# Test case 1: boxes intersect
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

print("iou for intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

# Test case 2: boxes do not intersect
box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection must be 0"

# Test case 3: boxes intersect at vertices only
box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

# Test case 4: boxes intersect at edge only
box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at edges must be 0"


def non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')    

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=max_boxes, iou_threshold=iou_threshold)
    
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    # Convert YOLO box predictions to bounding box corners.
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def scale_boxes (boxes, image_shape):
    
    orig_size = np.array([image_shape[0], image_shape[1]])
    orig_size = np.expand_dims(orig_size, axis=0)
    boxes = boxes.reshape((-1, 5))
  
    boxes_extents = boxes[:, [2, 1, 4, 3, 0]]

    # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
    boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
    boxes_xy = boxes_xy / orig_size
    boxes_wh = boxes_wh / orig_size
    boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)
    detectors_mask, matching_true_boxes = preprocess_true_boxes(boxes, anchors,
                                                                [416, 416])


def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    scores, boxes, classes = filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


#reading from class path

classes_path = "YAD2K/model_data/coco_classes.txt"
with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

#deciding anchors

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

anchors_path = "model_data/yolo_anchors.txt"

if os.path.isfile(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
else:
    anchors = YOLO_ANCHORS

model_image_size = (608, 608) # Same as yolo_model input layer size


yolo_model = load_model("YAD2K/yad2k/models/keras_yolo.py", compile=False)


yolo_model.summary()




def predict(image_file):

    # Preprocess
    image = "images/" + image_file 
    image = image.resize((608, 608), PIL.Image.BICUBIC)
    image_data = np.array(image, dtype=np.float)
    image_data /= 255.
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    #prediction output
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
  
    colors = get_colors_for_classes(len(class_names))
 
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    image.save(os.path.join("out", image_file), quality=100)
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes

