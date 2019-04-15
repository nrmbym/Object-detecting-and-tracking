# -*- coding: utf-8 -*
"""
--------------------------------------------------------------------
File        : VideoProcess.py
Time  		: 2019-4-14
Author      : Amo
Description : 视频流处理
Update      :
--------------------------------------------------------------------
"""

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
"""
视频流处理
"""


class VideoProcess(object):
    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = np.array([])
        self.originalFrame = np.array([])

    def capture_frame(self):
        """
        捕获帧
        :param  : None
        :return : Frame
        """
        ret, Frame = self.capture.read()
        return Frame

    def capture_next_frame(self):
        """
        捕获下一帧
        :param  : None
        :return : None
        """
        ret, self.originalFrame = self.capture.read()
        if ret is True:
            self.currentFrame = cv2.cvtColor(
                self.originalFrame, cv2.COLOR_BGR2RGB)

    def convert_frame(self):
        """
        帧格式转换
        :param  : None
        :return : img
        """
        try:
            height, width = self.currentFrame.shape[:2]
            # 用QImage加载当前帧
            # 需要注意QImage参数类型
            img = QImage(
                np.ndarray.tobytes(self.currentFrame),
                int(width),
                int(height),
                QImage.Format_RGB888)
            img = QPixmap.fromImage(img)
            return img
        except BaseException:
            return None
