# -*- coding: utf-8 -*
"""
--------------------------------------------------------------------
File        : DetectTarget.py
Author      : Amo
Time  		: 2019-3-26
Description : 小车检测主函数
Update      :   -- 2019/4/11
                    1.优化了class MainUi 中的方法
                    2.提取出了带检测目标的中心坐标
                    3.下一步进行特定目标的跟踪

                -- 2019/4/12
                    #################### 21:05 ####################
                    1.纠正错误思路:目标追踪不能一直做检测（效率太低），
                    K 帧(K = k1 + (K - k1)) 图像，需要检测 k1 帧，
                    剩下的 K - k1 帧由跟踪算法进行预测！

                -- 2019/4/14
                    #################### 20:20 ####################
                    修改思路：1.（选定）视频加载完成后截取一帧进行检测，选择"person"为目标
                             2.（跟踪）选定目标后用OpenCV的"CSRT"算法跟踪
                             3.（寻回）丢失目标后重新进行检测

                    #################### 22:42 ####################
                    移植没有完成，在self.tracker.init报错！！！
                    pedestrian_detect_track() 逻辑结构不够明了，功能实现后进行重写

                -- 2019/4/15
                    #################### 10:01 ####################
                    1.将 class VideoProcess 和 class GeneralYolov3 分离
                    2.在 github/Object-detect 创建新分支 dev
                    #################### 10:28 ####################
                    3.跟踪功能实现（之前 self.tracker.init 报错是因为 self.initBB 传参错误造成）
                    4.在 pedestrian_detect_track() 中将坐标发送加上
                    5.速度太慢，可以尝试优化，寻回功能还没写
                    #################### 15:55 ####################
                    6.send_msg() 优化，整体功能基本实现

--------------------------------------------------------------------
"""

import sys
import cv2
import time
from imutils.video import FPS
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from UI import DetectUi
from VideoProcess import VideoProcess
from GeneralYolov3 import GeneralYolov3

MODIFY = True

DETECTORTRACK = True  # 检测 or 跟踪
TINYORNOT = True  # 使用YOLOv3-Tiny?

YOLOV3PATH = "D:/Py/YOLOv3/"  # YOLO v3 配置文件目录和权重目录

# OpenCV 目标跟踪工具
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "goturn": cv2.TrackerGOTURN_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


# class MainUI 主窗口
class MainUi(DetectUi):
    def __init__(self):
        super().__init__()
        self.tic = time.time()

        # 目标检测/跟踪标志位
        # True--> 检测
        # False--> 跟踪
        self.dect_trac = DETECTORTRACK

        # initialize the bounding box coordinates of the object we are going to
        # track
        self.initBB = None

        # 选择 tracker
        self.tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

        # message 发送标志
        self.msg_count = 0

        # FPS
        self.fps = None

        # 构建神经网络对象
        self.yolov3 = GeneralYolov3(YOLOV3PATH, is_tiny=TINYORNOT)

        # 构建视频对象
        self.video = VideoProcess(cv2.VideoCapture(0))
        time.sleep(1)

        # 构建定时器
        self.timer = QTimer(self)
        # 设定定时器溢出时间
        self.timer.start(33)

        # 定时器溢出时发送信号
        self.timer.timeout.connect(self.load_video)

        # 开始加载视频
        self.load_video()

    def load_video(self):
        """
        视频加载函数
        :param  : None
        :return : None
        """
        self.msg_count += 1
        self.video.capture_next_frame()
        self.tic = time.time()

        if MODIFY:
            # YOLOV3 + Tracker
            self.pedestrian_detect_track()
        else:
            # YOLOV3
            self.pedestrian_detect_yolo()

        self.videoFrame.setPixmap(self.video.convert_frame())
        self.videoFrame.setScaledContents(True)

        print("%.4f" % (time.time() - self.tic))

    ##############################################################
    #                   TODO: YOLO 检测                          #
    #                   2019/3/29                               #
    ##############################################################
    def pedestrian_detect_yolo(self):
        """
        使用 YOLO v3 进行行人识别
        :return:
        """
        results, res_find = self.predict()
        if res_find:
            label = "正在跟踪目标...\n坐标:"
            # self.draw_speed()
            coordinates = self.processing_bounding_box(results)
            res_coordinate = self.compute_center(coordinates)
        else:
            label = "没有找到目标，正在重新寻找..."
            res_coordinate = None

        self.send_msg(finded=res_find,
                      coordinate=res_coordinate,
                      text=label)

    def predict(self):
        # 创建网络输入的 4D blob.
        # blobFromImage()将对图像进行预处理
        #       - 平均减法（用于帮助对抗数据集中输入图像的光照变化）
        #       - 按某种因素进行缩放
        blob = cv2.dnn.blobFromImage(self.video.currentFrame.copy(),
                                     1.0 / 255,
                                     (self.yolov3.net_width,
                                      self.yolov3.net_height),
                                     [0, 0, 0],
                                     swapRB=False,
                                     crop=False)
        # 设置模型的输入 blob
        self.yolov3.yolov3_model.setInput(blob)

        # 前向计算
        outputs = self.yolov3.yolov3_model.forward(self.yolov3.outputs_names)
        # 后处理
        results = self.yolov3.postprocess(
            self.video.currentFrame.copy(), outputs)

        return results, len(results)

    def processing_bounding_box(self, results):
        """
        处理边界框

        left,top-------------
        |                   |
        |                   |
        |                   |
        |                   |
        |                   |
        |                   |
        ---------right,bottom

        :param results:
        :return: coordinate : 边界框中心坐标
        """

        coordinate = None
        coord_temp = {}

        for result in results:
            left, top, right, bottom = result["box"]
            cv2.rectangle(self.video.currentFrame, (left, top),
                          (right, bottom), (255, 178, 50), 3)

            # coord_temp["class_id"] = results["class_id"]
            coord_temp["coordinate"] = (left, top, right, bottom)

            coordinate = coord_temp["coordinate"]

        return coordinate

    def draw_speed(self):
        """
        计算速率信息                    #
        getPerfProfile() 函数返回模型的推断总时间以及
        每一网络层的耗时(in layersTimes)
        :return: None
        """
        t, _ = self.yolov3.yolov3_model.getPerfProfile()
        label = 'Inference time: %.2f ms' % \
                (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(self.video.currentFrame, label, (10, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (95, 255, 127))

    def send_msg(self, **kwargs):
        """
        向文本浏览框发送信息
        :param results:
        :return:
        """
        if (self.msg_count % 20 == 0) and kwargs["finded"]:
            self.msg_count = 0
            self.textBrowser.append("%s %s,%s" % (kwargs["text"],
                                                  kwargs["coordinate"][0][0],
                                                  kwargs["coordinate"][0][1]))
        elif self.msg_count % 20 == 0:
            self.msg_count = 0
            self.textBrowser.append("%s" % kwargs["text"])

    def compute_center(self, coordinates):
        """
        计算中心坐标
        :param coordinates: tuple 类型 （left, top, right, bottum)
        :return:
        """
        res_coord = []
        if len(coordinates):
            # for ind in coordinates:
            #     # 计算中心坐标
            #     center_H = int(
            #         (ind["coordinate"][0] + ind["coordinate"][2]) / 2)
            #     center_V = int(
            #         (ind["coordinate"][1] + ind["coordinate"][3]) / 2)

            center_H = int((coordinates[0] + coordinates[2]) / 2)

            center_V = int((coordinates[1] + coordinates[3]) / 2)

            res_coord.append((center_H, center_V))
            print(res_coord[0])

            cv2.circle(self.video.currentFrame, (center_H, center_V),
                       10, (255, 0, 0), -5)

        return res_coord
    ##############################################################
    #                           END                              #
    ##############################################################

    ##############################################################
    #                   TODO: YOLO 检测 + Tracker                #
    #                   2019/4/14                               #
    ##############################################################
    def pedestrian_detect_track(self):
        if self.dect_trac:
            if self.msg_count == 0:
                pass  # 加载第一帧
            else:
                # 检测
                results, res_find = self.predict()

                if res_find:  # 如果检测到目标
                    self.dect_trac = False

                    self.initBB = self.processing_bounding_box(results)

                    # self.initBB = (coordinates[0]["coordinate"])

                    self.tracker.init(
                        self.video.currentFrame.copy(), self.initBB)

                    self.fps = FPS().start()

                else:
                    self.send_msg(finded=False, text="没有找到目标，正在重新寻找...")
        else:
            # 跟踪
            if self.initBB is not None:
                # 捕捉目标的新边界坐标
                (success, box) = self.tracker.update(self.video.currentFrame)

                # 查看是否捕捉成功
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(self.video.currentFrame, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

                    res_coordinate = self.compute_center((x, y, x + w, y + h))

                    self.send_msg(finded=True,
                                  coordinate=res_coordinate,
                                  text="正在跟踪目标...\n坐标:")

                # 更新FPS
                self.fps.update()
                self.fps.stop()
        ##############################################################
        #                           END                              #
        ##############################################################


def main():
    app = QApplication(sys.argv)

    detect_target = MainUi()

    detect_target.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
