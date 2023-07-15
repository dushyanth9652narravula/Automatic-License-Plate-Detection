import numpy as np
import cv2
import os
import keras_ocr

# INPUT Settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Load YOLO Model

net = cv2.dnn.readNetFromONNX('./static/Model2/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Creating Pipeline for optical charecter Recognition

pipeline = keras_ocr.pipeline.Pipeline()

def get_detections(img,net):

    # Convert Image into Yolo Format

    image = img.copy()
    row,col,d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype = np.uint8)
    input_image[0:row,0:col] = image

    # Get Prediction from Yolo Model

    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB = True,crop = False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image,detections

def non_maximum_suppresion(input_image,detections):

    boxes = []

    confidences = []

    image_w,image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):

        row = detections[i]
        confidence = row[4]
        if confidence>0.4:
            class_score = row[-1]
            if class_score > 0.25 :

                cx,cy,w,h = row[0:4]

                left = int((cx-0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # Cleaning the arrays

    boxes_np = np.array(boxes).tolist()
    confidence_np = np.array(confidences).tolist()

    # Applying Non Maximum Suppression on Bounding Box

    index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidence_np,0.25,0.45)).flatten()
    
    return boxes_np,confidence_np,index

def OCR(image,bbox,filename):
    
    x,y,w,h = bbox
    
    roi = image[y:y+h,x:x+w]

    cv2.imwrite('./static/roi/{}'.format(filename),roi)

    predictions = pipeline.recognize([roi])
    
    if predictions[0] :
        
        text = predictions[0][0][0]
        
        return text
    else :
        
        return ""

# Drawing the Bounding Boxes

def get_drawings(boxes_np,confidence_np,image,index,filename):

    text_list = []
    
    for ind in index :
    
        x,y,w,h = boxes_np[ind]
        bb_conf = confidence_np[ind]
        conf_text = 'plate : {:.0f}%'.format(bb_conf*100)
        
        license_text = OCR(image,boxes_np[ind],filename)
        

        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

        text_list.append(license_text)
    
    return image,text_list

def yolo_predictions(img,net,filename):
    
    # Predictions

    # Step 1 : Getting Detections

    input_image,detections = get_detections(img,net)

    # Step 2 : Applying Non Maximum Suppresion Technique

    boxes_np,confidence_np,index = non_maximum_suppresion(input_image,detections )

    # Step 3 : Gettings Drawings 

    results,text_list = get_drawings(boxes_np,confidence_np,img,index ,filename)
    
    return results,text_list

def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list = yolo_predictions(image,net,filename)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    return text_list






