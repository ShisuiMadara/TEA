import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
import math

dictionary ={
                    'White':([0, 0, 146], [180, 34, 255]),
                    'Gray':([0, 0, 22], [180, 34, 146]),
                    'Light-red':([0,157, 25], [6,255,255]),
                    'Light-Pink':([0,0, 25], [6,157,255]),
                    'Orange':([6, 33, 168], [23, 255, 255]),
                    'Brown':([6, 33, 25], [23, 255, 168]),
                    'yellow':([23, 33, 25], [32, 255, 255]),
                    'Green':([32, 33, 25], [75, 255, 255]), 
                    'Blue-Green':([75, 33, 25], [90, 255, 255]), 
                    'Blue':([90,33, 45], [123, 255, 255]),
                    'Purple':([123, 112, 25], [155, 255, 255]),
                    'Light-Purple':([123, 33, 25], [155, 125, 255]),                   
                    'Pink':([155,34, 25], [180,225,255]),
                    'Deep-Pink':([175,0, 25], [180,157,255]),
                    'Deep-red':([175,157, 25], [180,255,255]),    
                    'black':([0, 0, 0], [180, 255, 26]),      
                    }  
    

confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 416       
inpHeight = 416    



parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu' or 'gpu'.")
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if(args.device == 'cpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
elif(args.device == 'gpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	ppm = 8.8
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6
	return speed
	

def getOutputsNames(net):
   
    layersNames = net.getLayerNames()
    print(net.getUnconnectedOutLayers())
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom, name):

    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf

    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    label += (' ' + name)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)

    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]


    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    cars = []

    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        name = ''
        cars.append([left,left+width, top, top+height])

        print(cars)

        cropped = frame[top:top+height, left:left+width]

        name = getColors(cropped)
        
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, name)
       

       
def getColors(image):

    if(image.size == 0) :
        return ''
    
    
    image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    dictionary ={
                    'White':([0, 0, 116], [57, 57, 255]),
                    'Red':([180,0,0], [255,255,255]),
                    'Light-red':([0,38, 56], [10,255,255]),
                    'orange':([10, 38, 71], [20, 255, 255]),
                    'yellow':([18, 28, 20], [33, 255, 255]),
                    'green':([36, 10, 33], [88, 255, 255]), 
                    'blue':([87,32, 17], [120, 255, 255]),
                    'purple':([138, 66, 39], [155, 255, 255]),
                    'Deep-red':([170,112, 45], [180,255,255]),
                    'Deep-blue': ([110,100,100], [120,255,255]),
                    'black':([0, 0, 0], [179, 255, 50]),      
                    }  
    
    color_name = []
    color_count =[]
             

    for key,(lower,upper) in dictionary.items():
        
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
     
        mask = cv.inRange(image_HSV, lower, upper)
        
        count = cv.countNonZero(mask)
        
        color_count.append(count)
        
        color_name.append(key)
    
   
    color_count_array = np.array(color_count)
    
    idx = np.argmax(color_count_array)

    color = color_name[idx]

    
    return color



winName = 'VISSIM data extractor'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
   
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
  
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:

    cap = cv.VideoCapture(0)


if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    

    hasFrame, frame = cap.read()
    
   
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
  
        cap.release()
        break


    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob)


    outs = net.forward(getOutputsNames(net))

 
    postprocess(frame, outs)


    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
   

    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)

