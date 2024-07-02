from ultralytics import YOLO
import cv2
import math 
import depthai as dai
import time
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
# monoRight = pipeline.create(dai.node.MonoCamera)

xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
# xoutRight = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
xoutLeft.setStreamName("left")
# xoutRight.setStreamName("right")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
# monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)


# Linking
camRgb.video.link(xoutVideo.input)
monoLeft.out.link(xoutLeft.input)
# monoRight.out.link(xoutRight.input)

device= dai.Device(pipeline)
video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
left = device.getOutputQueue(name="left", maxSize=1, blocking=False)
# right = device.getOutputQueue(name="right", maxSize=1, blocking=False)



# model
model = YOLO("yolov8l.pt")

# object classes
classNames = ["person"]

# object details
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

#Initial brightness of IR led (range is 0 to 1)
ir_led=0
con=0.01

while True:
    videoIn = video.get()
    img=videoIn.getCvFrame()
    brightness_color=np.array(img).sum()
    
    results = model(img, stream=True)

    leftIn = left.get()
    imgl=leftIn.getCvFrame()
    brightness_ir=np.array(imgl).sum()
    # print(brightness_ir)    
    imgl0=cv2.cvtColor(imgl, cv2.COLOR_GRAY2BGR)
    resultsl = model(imgl0, stream=True)
    
    # rightIn = right.get()
    # imgr=rightIn.getCvFrame()
    # imgr0=cv2.cvtColor(imgr, cv2.COLOR_GRAY2BGR)
    # resultsr = model(imgr0, stream=True)

    if brightness_color>1e8:
        imgs=[(img, results) ]
    else:
        imgs=[(imgl, resultsl)]

    if brightness_ir<5e7 and ir_led<0.9:
        ir_led=ir_led+con
        device.setIrFloodLightIntensity(ir_led)
    elif brightness_ir>7e7 and ir_led>con:
        ir_led=ir_led-con
        device.setIrFloodLightIntensity(ir_led)

    print(ir_led)

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

    
    if brightness_color>1e8:
        img_view=img
    else:
        img_view=imgl

    cv2.imshow('Oak-D', img_view)


    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()

