from collections import defaultdict

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920, 1080)

# #For wide camera
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
# camRgb.setVideoSize(1280, 800)


xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)

# Connect to device and start pipeline
device= dai.Device(pipeline)
video = device.getOutputQueue(name="video", maxSize=1, blocking=False)



# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the YOLO model with segmentation capabilities
model = YOLO("yolov8n-seg.pt")


# # Retrieve video properties: width, height, and frames per second
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video with the specified properties
# out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    # Read a frame from the video
    videoIn = video.get()
    im0=videoIn.getCvFrame()
    


    # Perform object tracking on the current frame
    results = model.track(im0, persist=True)
    # print(results[0].boxes.id)
    
    for r in results:
            
        alpha=(r.boxes.cls==0).nonzero()
        if len(alpha)>0 and r.boxes.id is not None and r.masks is not None:
            for mask, box in zip (r.masks[alpha], r.boxes[alpha]):  

                # Create an annotator object to draw on the frame
                annotator = Annotator(im0, line_width=2)
    #         # Extract masks and tracking IDs
                masks = mask.xy
                track_ids = box.id.int().cpu().tolist()
                # # Annotate each mask with its corresponding tracking ID and color
                for mask, track_id in zip(masks, track_ids):
                    annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))

    im0=cv2.resize(im0, (640, int(640/1.8)), cv2.INTER_AREA)

    cv2.imshow("instance-segmentation-object-tracking", im0)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video writer and capture objects, and close all OpenCV windows
# out.release()
# cap.release()
cv2.destroyAllWindows()