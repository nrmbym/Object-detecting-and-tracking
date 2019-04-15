# -*- coding: utf-8 -*
"""
--------------------------------------------------------------------
File        : GeneralYlolov3.py
Time  		: 2019-4-14
Author      : Amo
Description : Yolo v3 模型加载、预测
Update      :
--------------------------------------------------------------------
"""

import cv2
import numpy as np
import os

COCONAMEPATH = "D:/Py/Car/coco.names"  # CoCo数据集分类类名

"""
class GeneralYolove 用于：
 - 设置网络参数
 - 加载 YOLO v3 模型
 - 加载 CoCo 分类类名
 - 处理分类结果
"""


class GeneralYolov3(object):
    def __init__(self, modelpath, is_tiny=False):
        self.conf_threshold = 0.5  # 置信度阈值
        self.nms_threshold = 0.4  # 非最大值抑制阈值
        self.net_width = 416  # 网络输入图像宽度
        self.net_height = 416  # 网络输入图像高度

        self.classes = self.get_coco_names()
        self.yolov3_model = self.get_yolov3_model(modelpath, is_tiny)
        self.outputs_names = self.get_outputs_names()

    def get_coco_names(self):
        """
        获取COCO数据集标签
        :return: classes : coco 分类标签
        """
        # COCO 物体类别名
        classes = None
        with open(COCONAMEPATH, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
            print("[INFO] COCO.names loaded ...")

        return classes

    def get_yolov3_model(self, modelpath, is_tiny):
        """
        :param modelpath: 模型路径
        :param is_tiny: 加载 YOLOv3-tiny 模型或 YOLOv3 模型
        :return: net : 返回网络
        """
        if is_tiny:
            cfg_file = os.path.join(modelpath, "yolov3-tiny.cfg")
            weights_file = os.path.join(modelpath, "yolov3-tiny.weights")
            print("[INFO] YOLOV3-tiny model loaded ...")
        else:
            cfg_file = os.path.join(modelpath, "yolov3.cfg")
            weights_file = os.path.join(modelpath, "yolov3.weights")
            print("[INFO] YOLOV3 model loaded ...")

        net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return net

    def get_outputs_names(self):
        """
        获取所有网络层名
        :return:
        """
        layersNames = self.yolov3_model.getLayerNames()
        print("[INFO] LayersNames loaded ...")
        # 输出网络层名，如无连接输出的网络层.
        return [layersNames[i[0] - 1] for i in
                self.yolov3_model.getUnconnectedOutLayers()]

    def postprocess(self, img_cv2, outputs):
        """
        检测结果后处理
        :param img_cv2: 输入图像
        :param outputs: 前向计算输出
        :return: results : 返回结果
        """
        # 采用 NMS 移除低 confidence 的边界框
        img_height, img_width, _ = img_cv2.shape

        # 只保留高 confidence scores 的输出边界框
        # 将最高 score 的类别标签作为边界框的类别标签
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)  # 返回索引值中最大值的id
                # 区分出 "person"
                if class_id == 0:
                    confidence = scores[class_id]
                    # 置信度大于阈值
                    if confidence > self.conf_threshold:
                        center_x = int(detection[0] * img_width)
                        center_y = int(detection[1] * img_height)
                        width = int(detection[2] * img_width)
                        height = int(detection[3] * img_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

        # NMS 处理， 消除 lower confidences 的冗余重叠边界框
        indices = cv2.dnn.NMSBoxes(boxes,
                                   confidences,
                                   self.conf_threshold,
                                   self.nms_threshold)
        results = []
        for ind in indices:
            res_box = {}
            res_box["class_id"] = class_ids[ind[0]]
            res_box["score"] = confidences[ind[0]]

            box = boxes[ind[0]]
            res_box["box"] = (box[0],
                              box[1],
                              box[0] + box[2],
                              box[1] + box[3])
            results.append(res_box)

        return results
