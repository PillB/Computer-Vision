from ultralytics import YOLO
import cv2

# load models
coco_model = YOLO('yolov8n.pt') # car detector model pretrain w COCO dataset
#license_plate_detector = YOLO('path to trained license plate model')

# load video
cap = cv2.VideoCapture('./samples/sample1.mp4')

vehicles = [2, 3, 5, 7,]
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1 # used to limit the frames for testing in dev
    ret, frame = cap.read()
    if ret and frame_nmr < 10:
        # detect vehicles
        detections = coco_model(frame)[0] #[0] to do 1 step at a time
        detections_ = [] #list where we will save all the bounding boxes our model detects
        for detection in detections.boxes.data.toList():
            x1, y1, x2, y2, score, class_id = detection #score is the confidence in the prediction, the class_id is the type of object
            if int(class_id) in vehicles:
                pass