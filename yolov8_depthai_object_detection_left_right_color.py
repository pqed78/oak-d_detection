from ultralytics import YOLO
import cv2
import math 
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)



xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)



# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setVideoSize(1920, 1080)

# #For wide camera
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
# camRgb.setVideoSize(1280, 800)


# xoutVideo.input.setBlocking(False)
# xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)
monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

device= dai.Device(pipeline)
video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
left = device.getOutputQueue(name="left", maxSize=1, blocking=False)
right = device.getOutputQueue(name="right", maxSize=1, blocking=False)



# model
model = YOLO("yolov8s.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# object details
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

while True:
    videoIn = video.get()
    img=videoIn.getCvFrame()
    results = model(img, stream=True)

    leftIn = left.get()
    imgl=leftIn.getCvFrame()
    imgl0=cv2.cvtColor(imgl, cv2.COLOR_GRAY2BGR)
    resultsl = model(imgl0, stream=True)
    
    rightIn = right.get()
    imgr=rightIn.getCvFrame()
    imgr0=cv2.cvtColor(imgr, cv2.COLOR_GRAY2BGR)
    resultsr = model(imgr0, stream=True)

    # imgs=[(img, results),(imgl, resultsl), (imgr, resultsr) ] # Color image is not working this case
    imgs=[(imgl, resultsl), (imgr, resultsr), (img, results) ]

    for img, results in imgs:
        # coordinates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # class name
                cls = int(box.cls[0])
                if cls==0:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                    org = [x1, y1]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Oak-D', img)
    cv2.imshow('Oak-D_left', imgl)
    cv2.imshow('Oak-D_Right', imgr)

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk