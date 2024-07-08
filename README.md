# YoloV8 object detection, segmentation, tracking using Oak-D PRO for the depth extraction with the Ultralytics APIs (detect person only)
- yolov8_depthai_object_Detection.py: object detection only with color camera

- yolov8_depthai_object_detection_left_right_color.py: object detection with all three cameras

- yolov8_depthai_object_detection_left_color_brightness.py: Depending on the overall brightness, camera used to detect objects is changed from color to mono and vice versa. IR led control added.

- yolov8_depthai_object_detection_left_color_brightness_distance.py: Depending on the overall brightness, camera used to detect object is changed from color to mono and vice versa with the distance in the surface normal direction between the camera and object. 
- yolov8_object_tracking_and_segmentation.py: object(only person) tracking and segmentation with oak-D (only color)


# Show the depth of ROI in RGB camera
-spatial_location_and_color_calculator.py: calculate distance between the camera and object, shown in color camera

# Pip list (FYI)
- requirement.txt
