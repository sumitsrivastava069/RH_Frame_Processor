import cv2
import numpy as np
import os
from datetime import datetime
import time
import threading
from flask import Flask
import boto3
from io import BytesIO
from PIL import Image
import random

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# Import Load Yolo Model 
# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#function get detection
def get_detections(img,net):
  image = img.copy()
  row, col, d = image.shape
  max_rc = max(row,col)
  input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
  input_image[0:row,0:col] = image
  ##get prediction from yolo model
  blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
  net.setInput(blob)
  preds = net.forward()
  detections = preds[0]
  return input_image, detections


def non_max_supression(input_image,detections):
  boxes = []
  confidences = []
  image_w, image_h = input_image.shape[:2]
  x_factor = image_w/INPUT_WIDTH
  y_factor = image_h/INPUT_HEIGHT
    ####

  for i in range(len(detections)):
    row = detections[i]
    confidence = row[4] # confidence of detecting license plate
    if confidence > 0.4:
      class_score = row[5] # probability score of license plate
      if class_score > 0.25:
        cx, cy , w, h = row[0:4]
        left = int((cx - 0.5*w)*x_factor)
        top = int((cy-0.5*h)*y_factor)
        width = int(w*x_factor)
        height = int(h*y_factor)
        box = np.array([left,top,width,height])
        confidences.append(confidence)
        boxes.append(box)


  # clean
  boxes_np = np.array(boxes).tolist()
  confidences_np = np.array(confidences).tolist()
  # NMS
  index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
  return boxes_np, confidences_np, index


def drawings(image, boxes_np,confidences_np,index):  
    ##drawing 

  for ind in index:
    x,y,w,h =  boxes_np[ind]
    bb_conf = confidences_np[ind]
    conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
    license_text = extract_text(image,boxes_np[ind])
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
    cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)
    cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
    cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
  return image



csv1_file = './numberplateimages/numberplates_data.csv'


def write_to_csv(frame_name, image_name, timestamp, number_plate_text):
    with open(csv1_file, 'r') as file:
        reader = csv.reader(file)
        existing_entries = set(row[2] for row in reader)  # Assuming number plate is in the third column

    if number_plate_text not in existing_entries:
        color = random.choice(['Green', 'Red'])
        with open(csv1_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_name, image_name, timestamp, number_plate_text, color])


def aws_textract(your_numpy_array):
    img = Image.fromarray(np.uint8(your_numpy_array))

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    client = session.client('textract', region_name='ap-south-1')
    response = client.detect_document_text(Document={'Bytes': img_byte_arr.read()})

    blocks = response['Blocks']
    if len(blocks) > 1:
        return blocks[1]['Text']
    else:
        return ''

def compare_number_plates(current_number_plate):
    with open(csv1_file, 'r') as file:
        reader = csv.reader(file)
        existing_number_plates = set()
        for plate in reader:
            if len(plate) > 2:
                existing_number_plates.add(plate[2])  # Assuming the number plate is in the third column
        if current_number_plate in existing_number_plates:
            return True
    return False


def update_csv(image_path, timestamp, number_plate_text):
    filename = os.path.basename(image_path)

    with open(csv1_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, timestamp, number_plate_text])
##Not comparing
numberis = []
numberis.append('up53aj9382')
access_key='AKIASVXC3DJ65DYMMWQQ'
secret_key='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'

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return ''
    
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    text = str(aws_textract(gray))
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    
    
    if not compare_number_plates(text):
        plate_filename = f"./numberplateimages/plate_{timestamp}.jpg"
        yolo_filename = f"./yoloimages/vehicle_{timestamp}.jpg"
        cv2.imwrite(yolo_filename, image)
        cv2.imwrite(plate_filename, gray)
        write_to_csv(yolo_filename,plate_filename, timestamp, text)
        print(text)


  


##combinin all the functions
def yolo_prediction(img,net):
    ##Step1 - prediction
  input_image, detections = get_detections(img,net)
    ##step2 - NMS
  boxes_np, confidences_np, index = non_max_supression(input_image,detections)
    ##Step3 - Drawings
  results_img = drawings(img, boxes_np,confidences_np,index)
  return results_img














from flask import Flask
import cv2
import os
import time
import threading
import csv

app = Flask(__name__)

# Specify the folder path
folder_path = './frameprocessorstorage/'
last_processed_time = 0
csv_file = './frameprocessorstorage/processed_frames.csv'
processed_filenames = set()

def load_processed_filenames():
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 0:
                    filename = row[0]
                    processed_filenames.add(filename)

def update_csv(image_path, timestamp):
    filename = os.path.basename(image_path)
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, timestamp])

def process_images():
    global last_processed_time

    load_processed_filenames()

    while True:
        file_list = os.listdir(folder_path)
        filtered_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg')) and os.path.getmtime(os.path.join(folder_path, f)) > last_processed_time]
        sorted_files = sorted(filtered_files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

        for latest_image in sorted_files:
            image_path = os.path.join(folder_path, latest_image)
            if latest_image not in processed_filenames:
                print("Processing image:", latest_image)
                image = cv2.imread(image_path)
                # Perform image processing
                results = yolo_prediction(image, net)

                # Get the timestamp
                timestamp = os.path.getmtime(image_path)

                # Update CSV
                update_csv(image_path, timestamp)

                last_processed_time = os.path.getmtime(image_path)
                processed_filenames.add(latest_image)  # Add the processed image filename to the set

        time.sleep(5)

# Run the image processing thread
image_processing_thread = threading.Thread(target=process_images)
image_processing_thread.daemon = True
image_processing_thread.start()

# Run the Flask app
if __name__ == '__main__':
    app.run()
