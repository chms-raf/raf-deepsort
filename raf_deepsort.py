

from detectron2.utils.logger import setup_logger
setup_logger()

from Xlib import X, display
from collections import namedtuple

# import some common libraries
import numpy as np
import cv2

import rospy
import sys
from cv_bridge import CvBridge
from raf.msg import Result
from raf.msg import DetectionList, RafState
from sensor_msgs.msg import Image, RegionOfInterest


from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes

class updateClasses(object):
    def __init__(self):
        # Params
        self.arm_image = None
        self.scene_image = None
        self.image = None
        self.raf_state = None
        self.detections = None
        self.br = CvBridge()

        self.detectron2 = Detectron2(detectron2_checkpoint="/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/final_model/model_final.pth", use_cuda="True")
        self.deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7", use_cuda="True")

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('sorted_detections', DetectionList, queue_size=10)

        # Subscribers
        rospy.Subscriber("/raf_state", RafState, self.state_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.arm_callback)
        rospy.Subscriber("/scene_camera/color/image_raw", Image, self.scene_callback)
        rospy.Subscriber("/arm_camera_detections", DetectionList, self.detection_callback)

    def arm_callback(self, msg):
        self.arm_image = self.convert_to_cv_image(msg)

    def scene_callback(self, msg):
        self.scene_image = self.convert_to_cv_image(msg)

    def state_callback(self, msg):
        self.raf_state = msg
        
    def detection_callback(self, msg):
        self.detections = msg

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    def publish(self, det):
        self.pub.publish(det)
        self.loop_rate.sleep()

def main():
    """ Test """
    rospy.init_node("raf_deepsort", anonymous=True)
    bridge = CvBridge()

    run = updateClasses()

    while not rospy.is_shutdown():

        if run.raf_state is None:
            continue

        # Check which view is selected
        if run.raf_state.view == "arm" and run.raf_state.enable_arm_detections:
            run.image = run.arm_image
        elif run.raf_state.view == 'scene' and run.raf_state.enable_scene_detections:
            run.image = run.scene_image

        # If no image or no detections, return to start of loop
        if run.image is None or run.detections is None:
            continue

        # Get detections
        detections = run.detections

        # Map boxes to right format
        bbox_xcycwh = list()
        for box in detections.boxes:
            bbox_xcycwh.append([box.x_offset, box.y_offset, box.height, box.width])

        # TODO: Loop through classes I care about
        # select class cup
        temp = detections.class_names == 'Cup'

        # bbox_xcycwh = bbox_xcycwh[temp]
        bbox_xcycwh = np.array(bbox_xcycwh, dtype=np.float64)
        # scores = detections.scores[temp]
        scores = np.array(detections.scores, dtype=np.float32)
        # bbox_xcycwh[:, 3:] *= 1.2   # Widen the bbox by 20%

        # DeepSort
        # TODO: bbox_xcycwh is the wrong format
        outputs = run.deepsort.update(bbox_xcycwh, scores, run.image)

        print(outputs)

        # sorted_bbox_xyxy = outputs[:, :4]

        # # Create message for publishing
        # # TODO: Need to get classes beyond just cups right now
        # sorted_detections = DetectionList()
        # class_ids = list()
        # class_names = list()
        # scores = list()
        # masks = list()
        # for i in range(len(outputs)):
        #     box = RegionOfInterest()
        #     box.x_offset = np.uint32(bbox_xcycwh[i, 0])
        #     box.y_offset = np.uint32(bbox_xcycwh[i, 1])
        #     box.height = np.uint32(bbox_xcycwh[i, 2])
        #     box.width = np.uint32(bbox_xcycwh[i, 3])
        #     box.do_rectify = False
        #     result_msg.boxes.append(result_ROI)
        #     result_msg.class_ids.append(np.int32(2))
        #     result_msg.class_names.append(class_names[2])

        # # TODO: Have deepsort return the cls_conf so I can display it back on the original image (not 100% necessary)
        # outputs = run.deepsort.update(bbox_xcycwh, cls_conf, img)
        # if len(outputs) <= 0:
        #     continue

        # bbox_xyxy = outputs[:, :4]

        # Create copies of the original image
        # im = img.copy()
        # im = draw_bboxes(im, bbox_xyxy, outputs[:, -1])  
        
    return 0

if __name__ == '__main__':
    # Main program
    sys.exit(main())