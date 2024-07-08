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
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutVideo.setStreamName("video")
xoutLeft.setStreamName("left")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# color properties

camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

#mono properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setCamera("right")

#stereo properties
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY) # 필요시 HIGH_ACCURACY 사용
# stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) #original is CAM_A:rgb_camera, centered Depth카메라와 중앙카depth메라 Align
# stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 300 #minimum distance in mm
config.depthThresholds.upperThreshold = 7000 #maximum distance in mm
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
# config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)
depth_area_scale=0.5


# Linking
camRgb.video.link(xoutVideo.input)
monoLeft.out.link(xoutLeft.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)


spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)



device= dai.Device(pipeline)
video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")


# model
model = YOLO("yolov8n.pt")

# object classes
classNames = ["person"]

# object details
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
color = (255, 255, 0)
thickness = 3
gap=100

#Initial brightness of IR led (range is 0 to 1)
ir_led=0
var=0.01


while True:
    videoIn = video.get()
    img=videoIn.getCvFrame()
    # print(img.shape)
    brightness_color=np.array(img).sum()
    
    results = model(img, stream=True)

    leftIn = left.get()
    imgl=leftIn.getCvFrame()
    # print(imgl.shape)
    brightness_ir=np.array(imgl).sum()
    # print(brightness_ir)    
    imgl0=cv2.cvtColor(imgl, cv2.COLOR_GRAY2BGR)
    resultsl = model(imgl0, stream=True)
    
    inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
    depthFrame = inDepth.getFrame() # depthFrame values are in millimeters
    # print(depthFrame.shape)

    # depth_downscaled = depthFrame[::4]
    # if np.all(depth_downscaled == 0):
    #     min_depth = 0  # Set a default minimum depth value when all elements are zero
    # else:
    #     min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
    # max_depth = np.percentile(depth_downscaled, 99)
    # depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
    # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
    spatialData = spatialCalcQueue.get().getSpatialLocations()


    
    if brightness_color>1e8:
        imgs=[(img, results)]
        device.setIrLaserDotProjectorBrightness(100)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    else:
        imgs=[(imgl, resultsl)]
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        device.setIrLaserDotProjectorBrightness(0)

    if brightness_ir<1e8 and ir_led<0.9:
        ir_led=ir_led+var
        device.setIrFloodLightIntensity(ir_led)
    elif brightness_ir>5e7 and ir_led>(var-0.001):
        ir_led=ir_led-var
        device.setIrFloodLightIntensity(ir_led)

    if np.abs(ir_led)<0.001:
        ir_led=0
        device.setIrFloodLightIntensity(ir_led)

    # print(ir_led)

    

    

    for img, results in imgs:

        for r in results:
            alpha=(r.boxes.cls==0).nonzero()
            k=0
            if len(alpha)>0:
                for i in range(len(alpha)):
                    if np.array(r.boxes.xywh[alpha][i][0][2:]).sum()>k:
                        k=np.array(r.boxes.xywh[alpha][i][0][2:]).sum()
                        index=i
                    
                zmin=50

                # print(alpha[0])
                for box in r.boxes[alpha[index]]:

                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                                            
                    
                    x10, y10, x20, y20 = box.xyxyn[0]

                    topLeft = dai.Point2f((x10+x20)/2+(x10-x20)/2*depth_area_scale, (y10+y20)/2+(y10-y20)/2*depth_area_scale)
                    bottomRight = dai.Point2f((x10+x20)/2-(x10-x20)/2*depth_area_scale, (y10+y20)/2-(y10-y20)/2*depth_area_scale)
                    
                    cfg = dai.SpatialLocationCalculatorConfig()
                    config.roi = dai.Rect(topLeft, bottomRight)
                    cfg.addROI(config)
                    spatialCalcConfigInQueue.send(cfg)
                    spatialData = spatialCalcQueue.get().getSpatialLocations()
                    # time.sleep(0.1)



                    # if zmin==50 or int(spatialData[0].spatialCoordinates.z)<zmin:
                    # if zmin==50:
                    xmin=x1
                    xmax=x2
                    ymin=y1
                    ymax=y2
                    xcen=int(spatialData[0].spatialCoordinates.x)
                    ycen=int(spatialData[0].spatialCoordinates.y)
                    zmin=int(spatialData[0].spatialCoordinates.z)
                
                # time.sleep(0.1) 
        # print(img.shape[0])
                    if img.shape[0]==720:
                        fontScale2=fontScale/2
                        thickness2=thickness
                        # gap2=gap/3
                    elif img.shape[0]==1080:
                        fontScale2=fontScale
                        thickness2=thickness
                        # gap2=gap

                
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)    
                    cv2.putText(img, classNames[0], (xmin, ymin-int(fontScale2*30)), font, int(fontScale2),color, int(thickness2))
                    cv2.putText(img, f"X: {xcen} mm", (xmin + 10, ymin + int(fontScale2*30)), font, int(fontScale2), color, int(thickness2))
                    cv2.putText(img, f"Y: {ycen} mm", (xmin + 10, ymin + int(2*fontScale2*30)), font, int(fontScale2), color, int(thickness2))
                    cv2.putText(img, f"Z: {zmin} mm", (xmin + 10, ymin + int(3*fontScale2*30)), font, int(fontScale2), color, int(thickness2))


    
    if brightness_color>1e7:
        img_view=img
    else:
        img_view=imgl
    img_view=cv2.resize(img_view, (640, int(640/1.8)), cv2.INTER_AREA)
    
    cv2.imshow('Oak-D', img_view)
    # cv2.imshow('Oak-D2', depthFrameColor)


    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()

