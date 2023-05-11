from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from cv_bridge import CvBridge

class Detectron2:

    def __init__(self, detectron2_checkpoint=None, use_cuda=True):
        self.cfg = get_cfg()
        # self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # BEGIN ADDED CODE
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # END ADDED CODE
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
        # self.cfg.MODEL.WEIGHTS = detectron2_checkpoint if detectron2_checkpoint else "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        # BEGIN ADDED CODE
        # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/output/model_final.pth")
        self.cfg.MODEL.WEIGHTS = detectron2_checkpoint
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
        # END ADDED CODE
        if not use_cuda: self.cfg.MODEL.DEVICE='cpu'
        self.predictor = DefaultPredictor(self.cfg)
        self.br = CvBridge()

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        outputs = self.predictor(im)
        # boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        # classes = outputs["instances"].pred_classes.cpu().numpy()
        # scores = outputs["instances"].scores.cpu().numpy()
        # masks = outputs["instances"].pred_masks.cpu().numpy()

        predictions = outputs["instances"].to("cpu")
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        scores = predictions.scores if predictions.has("scores") else None
        # masks = np.asarray(predictions.pred_masks) if predictions.has("masks") else None
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)

        bbox_xcycwh, cls_conf, cls_ids, mask_array = [], [], [], []

        for (box, _class, score) in zip(boxes, classes, scores):

            # if _class == 0:
            x0, y0, x1, y1 = box
            bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            cls_conf.append(score)
            cls_ids.append(_class)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.br.cv2_to_imgmsg(mask)
            mask_array.append(mask)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), mask_array
