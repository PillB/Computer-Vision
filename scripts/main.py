from ultralytics import YOLO
import cv2

# load models
coco_model = YOLO('yolov8n.pt') # car detector model pretrain w COCO dataset
license_plate_detector = YOLO('path to trained license plate model')

# load video
cv2.VideoCapture('./samples/sample1.mp4')

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1 # used to limit the frames for testing in dev
    ret, frame = cap.read()
    if ret and frame_nmr < 10:
        # detect vehicles
        detections = coco_model(frame)[0] #[0] to do 1 step at a time
        print(detections)