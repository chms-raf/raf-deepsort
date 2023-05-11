# This code is the testbed for developing a way for user's to select
# which class an item with low confidence is supposed to be. We will
# then use one-shot learning to improve the model based on the user's
# feedback. This version includes deepSORT for detection tracking

# Author: Jack Schultz
# Email: jschultz299@gmail.com
# Created 3/14/22

from detectron2.utils.logger import setup_logger
setup_logger()

from Xlib import X, display
from collections import namedtuple

# import some common libraries
import numpy as np
import os, cv2
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import rospy
import sys
from cv_bridge import CvBridge
from raf.msg import Result
from sensor_msgs.msg import Image, RegionOfInterest


from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes

class Detection:
    def __init__(self, id, name, score, box, mask, x, y):
        self.id = id
        self.name = name
        self.score = score
        self.box = box
        self.mask = mask
        self.x = x
        self.y = y

    def __repr__(self):
        return repr([self.id, self.name, self.score, self.box, self.mask, self.x, self.y])

class updateClasses(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        self.detectron2 = Detectron2(detectron2_checkpoint="/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/final_model/model_final.pth", use_cuda="True")
        self.deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7", use_cuda="True")

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_objects', Image, queue_size=10)
        self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def get_img(self):
        result = self.image
        return result

    def getResult(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #print(type(masks))
        else:
            return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.br.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

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

    def publish(self, img, result_msg):
        self.pub.publish(img)
        if result_msg is not None:
            self.result_pub.publish(result_msg)
        self.loop_rate.sleep()

    def compute_updates(self, image, class_id, objects):
        # Based on clicked item, draw options

        # compute centroids
        centroids = list()
        # for i in [x for x in range(len(objects.class_ids)) if x != class_id]:
        for i in range(len(objects.class_ids)):
            x1 = objects.boxes[i].x_offset
            y1 = objects.boxes[i].y_offset
            x2 = x1 + objects.boxes[i].width
            y2 = y1 + objects.boxes[i].height
            xc = np.mean([x1,x2])
            yc = np.mean([y1,y2])
            centroids.append([xc,yc])

        centroids_to_avoid = list()
        for i in [x for x in range(len(objects.class_ids)) if x != class_id]:
            centroids_to_avoid.append(centroids[i])

        centroids_to_avoid = np.array(centroids_to_avoid)
        avoid = (int(np.mean(centroids_to_avoid[:,0])), int(np.mean(centroids_to_avoid[:,1])))
        print(avoid)
        return centroids, avoid

    def compute_masks(self, im, cls_ids, mask_array, colors):
        masks = []
        masks_indices = []
        for i in range(len(cls_ids)):
            # Obtain current object mask as a numpy array (black and white mask of single object)
            current_mask = self.br.imgmsg_to_cv2(mask_array[i])

            # Find current mask indices
            mask_indices = np.where(current_mask==255)

            # Add to mask indices list
            if len(masks_indices) > len(cls_ids):
                masks_indices = []
            else:
                masks_indices.append(mask_indices)

            # Add to mask list
            if len(masks) > len(cls_ids):
                masks = []
            else:
                masks.append(current_mask)

            # Select correct object color
            color = colors[cls_ids[i]]

            # Change the color of the current mask object
            im[masks_indices[i][0], masks_indices[i][1], :] = color

        return im

def main():
    """ Test """
    rospy.init_node("run_network_with_deepsort", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0
    
    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train/annotations.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_set")
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 1000 # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # 5 classes (Plate, Carrot, Celery, Pretzel, Gripper)

    # Temporary Solution. If I train again I think I can use the dynamically set path again
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/final_model/model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("test_set")
    predictor = DefaultPredictor(cfg)

    class_names = ['Plate', 'Bowl', 'Cup', 'Fork', 'Spoon', 'Knife', 'Pretzel', 'Carrot', 'Celery', 
                   'Strawberry', 'Banana', 'Watermelon', 'Yogurt', 'Cottage Cheese', 'Beans', 'Gripper']

    color_plate = [255, 221, 51]
    color_bowl = [83, 50, 250]
    color_cup = [183, 209, 52]
    color_fork = [124, 0, 255]
    color_spoon = [55, 96, 255]
    color_knife = [51, 255, 221]
    color_pretzel = [83, 179, 36]
    color_carrot = [245, 61, 184]
    color_celery = [102, 255, 102]
    color_strawberry = [250, 183, 50]
    color_banana = [51, 204, 255]
    color_watermelon = [112, 224, 131]
    color_yogurt = [55, 250, 250]
    color_cottage_cheese = [179, 134, 89]
    color_beans = [240, 120, 140]
    color_gripper = [80, 80, 178]   
    colors = list([color_plate, color_bowl, color_cup, color_fork, color_spoon, color_knife, 
                   color_pretzel, color_carrot, color_celery, color_strawberry, color_banana, 
                   color_watermelon, color_yogurt, color_cottage_cheese, color_beans, color_gripper])

    alpha = .4

    run = updateClasses()

    rospy.sleep(1)
    print("Running...")

    # Set Up Stuff for Cursor
    d = display.Display()
    s = d.screen()
    root = s.root
    # root.warp_pointer(100,540)    # Move cursor off of objects
    d.sync()

    while not rospy.is_shutdown():

        # Get images
        img = run.get_img()

        # If no image, return to start of loop
        if img is None:
            continue

        # Get detections and sort using DeepSort
        bbox_xcycwh, cls_conf, cls_ids, masks = run.detectron2.detect(img)

        # print("\n----------------------")
        # print(bbox_xcycwh.shape)
        # print(bbox_xcycwh)
        # print(cls_ids)   
        # print(names)  
        # print("----------------------\n")

        if bbox_xcycwh is None or len(bbox_xcycwh) < 1:
            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            run.publish(im_msg, None)
        else:
            # try:
            #     names = np.array(class_names)[cls_ids]
            # except:
            #     print("ERROR!!!!")
            #     print(cls_ids)

            # TODO: Loop through classes I care about
            # select class cup
            temp = cls_ids == 2
            bbox_xcycwh = bbox_xcycwh[temp]
            # bbox_xcycwh[:, 3:] *= 1.2   # Widen the bbox by 20%
            cls_conf = cls_conf[temp]

            # Create message for publishing
            # TODO: This needs to be customized with more data
            # TODO: Need to get classes beyond just cups right now
            result_msg = Result()
            for i in range(bbox_xcycwh.shape[0]):
                result_ROI = RegionOfInterest()
                result_ROI.x_offset = np.uint32(bbox_xcycwh[i, 0])
                result_ROI.y_offset = np.uint32(bbox_xcycwh[i, 1])
                result_ROI.height = np.uint32(bbox_xcycwh[i, 2])
                result_ROI.width = np.uint32(bbox_xcycwh[i, 3])
                result_ROI.do_rectify = False
                result_msg.boxes.append(result_ROI)
                result_msg.class_ids.append(np.int32(2))
                result_msg.class_names.append(class_names[2])

            # TODO: Have deepsort return the cls_conf so I can display it back on the original image (not 100% necessary)
            outputs = run.deepsort.update(bbox_xcycwh, cls_conf, img)
            if len(outputs) <= 0:
                continue

            bbox_xyxy = outputs[:, :4]

            # Create copies of the original image
            im = img.copy()
            im = draw_bboxes(im, bbox_xyxy, outputs[:, -1])


            ######## DRAW STUFF ##########

            # Compute Masks
            # im = run.compute_masks(im, cls_ids, masks, colors)

            # # Draw object masks on image
            # cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)

            # print("SHAPE: ")
            # print(bbox_xyxy.shape)

            # # Draw object bbox, class label, and score on image
            # for i in range(len(cls_ids)):
            #     # Draw Bounding boxes
            #     start_point = (bbox_xyxy[i][0], bbox_xyxy[i][1])
            #     end_point = (bbox_xyxy[i][2], bbox_xyxy[i][3])
            #     start_point2 = (bbox_xyxy[i][0] + 2, bbox_xyxy[i][1] + 2)
            #     end_point2 = (bbox_xyxy[i][2] - 2, bbox_xyxy[i][1] + 12)
            #     # color = colors[cls_ids[i]]
            #     color = [0, 0, 0]
            #     box_thickness =  1

            #     name = names[i]
            #     score = cls_conf[i]
            #     conf = round(score.item() * 100, 1)

            #     # Test strategy for updating model classes
            #     if score > .9:
            #         string = str(name) + ":" + str(conf) + "%"
            #     else:
            #         string = str(name) + "??? - " + str(conf) + "%"
                
            #     # TODO: if cursor is within object bbox, highlight yellow
            #     # TODO: if clicked, changed clicked to true

            #     # if cls_ids[i] == 0:
            #     #     clicked = True
            #     #     selected_id = i
            #     # else:
            #     #     clicked = False

            #     # if rel_cursor[0] is not None and rel_cursor[1] is not None:
            #     #     if rel_cursor[0] >= start_point[0] and rel_cursor[0] <= end_point[0] and rel_cursor[1] >= start_point[1] and rel_cursor[1] <= end_point[1]:
            #     #         color = [0, 220, 255] # yellow
            #     #         box_thickness = 2
            #     #         contain_id = i
            #     #     else:
            #     #         color = [0, 0, 0]
            #     #         contain_id = None
            #     #         box_thickness = 1

            #     #     if contain_id is not None and click == "Yes":
            #     #         # print("Object " + str(i) + " (" + str(name) + ") was clicked")
            #     #         rospy.sleep(.05) # prevent multiple clicks detected
            #     #         # centroids, avoid = run.compute_updates(output, contain_id, result)
            #     #         clicked = True
            #     #         selected_id = contain_id

            #     # Draw
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     org = (bbox_xyxy[i][0] + 2, bbox_xyxy[i][1] + 10)
            #     fontScale = .3
            #     text_thickness = 1
            #     output = cv2.rectangle(output, start_point, end_point, color, box_thickness)
            #     output = cv2.rectangle(output, start_point2, end_point2, color, -1)     # Text box
            #     output = cv2.putText(output, string, org, font, fontScale, [255, 255, 255], text_thickness, cv2.LINE_AA, False)
                ######## END DRAW STUFF ##########

            # if clicked:
            #     # Change selected bbox color to green
            #     output = cv2.rectangle(output, (bbox_xyxy[selected_id][0], bbox_xyxy[selected_id][1]), \
            #         (bbox_xyxy[selected_id][2], bbox_xyxy[selected_id][3]), \
            #         [0, 255, 0], 2)
                
            #     # Display options to the right of the bbox
            #     start_x = bbox_xyxy[selected_id][2] + 20
            #     start_y = bbox_xyxy[selected_id][1] - 20

            #     x = start_x
            #     y = start_y + 20

            #     for i in range(len(class_names)):
            #         y += 20
            #         output = cv2.rectangle(output, (x, y), (x+40, y+12), [255, 255, 255], -1)
            #         output = cv2.putText(output, class_names[i], (x+2, y+8), font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)

                # The below is potentially for a more robust way of displaying class update options
                # for i in range(len(centroids)):
                #     if i == selected_id:
                #         circle_color = [0, 0, 255]
                #     else:
                #         circle_color = [0, 255, 0]
                #     output = cv2.circle(output, (int(centroids[i][0]), int(centroids[i][1])), 2, circle_color, 2)
                # output = cv2.circle(output, avoid, 2, [255, 0, 0], 2)
                # output = cv2.arrowedLine(output, avoid, (int(centroids[selected_id][0]), int(centroids[selected_id][1])), [255,0,0], 1)  # y-axis (tag frame)

            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

            run.publish(im_msg, result_msg)    
        
    return 0

if __name__ == '__main__':
    # Main program
    sys.exit(main())