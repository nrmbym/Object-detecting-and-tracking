# -*- coding: utf-8 -*
"""
--------------------------------------------------------------------
File        : UI.py
Time  		: 2019-3-26
Author      : Amo
Description : 小车UI窗口
Update      :
--------------------------------------------------------------------
"""

from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QWidget, QTextBrowser, QLabel, QFrame


class DetectUi(QWidget):
    def __init__(self):
        super().__init__()
        # ——————————构建视频帧显示标签——————————— #
        self.videoFrame = QLabel(self)
        # ——————————构建一个文本浏览框——————————— #
        self.textBrowser = QTextBrowser(self)
        # ——————————————设置标签——————————————— #
        self.label_1 = QLabel(self)
        # ———————————————设置边框——————————————— #
        self.line_1 = QFrame(self)
        self.line_2 = QFrame(self)
        self.line_3 = QFrame(self)
        self.line_4 = QFrame(self)
        self.line_5 = QFrame(self)
        # ———————————————构建窗口——————————————— #
        self.setGeometry(100, 100, 625, 400)
        # 构建布局
        self.setupUi()

    def setupUi(self):
        """
        构建窗口布局
        :param  : None
        :return : None
        """
        # ———————————————设置字体——————————————— #
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)

        self.videoFrame.setGeometry(30, 70, 320, 240)
        self.videoFrame.setObjectName("videoFrame")

        self.textBrowser.setGeometry(370, 70, 220, 240)

        self.label_1.setGeometry(230, 30, 180, 20)
        self.label_1.setFont(font)
        self.label_1.setText("自动跟随智能购物车")

        self.line_1.setGeometry(30, 70, 320, 3)
        self.line_1.setFrameShape(QFrame.HLine)
        self.line_1.setFrameShadow(QFrame.Sunken)
        self.line_1.setObjectName("line_1")

        self.line_2.setGeometry(30, 310, 320, 3)
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.line_3.setGeometry(30, 70, 3, 240)
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)
        self.line_3.setObjectName("line_3")

        self.line_4.setGeometry(350, 70, 3, 240)
        self.line_4.setFrameShape(QFrame.VLine)
        self.line_4.setFrameShadow(QFrame.Sunken)
        self.line_4.setObjectName("line_4")

        self.line_5.setGeometry(360, 70, 3, 240)
        self.line_5.setFrameShape(QFrame.VLine)
        self.line_5.setFrameShadow(QFrame.Sunken)
        self.line_5.setObjectName("line_5")

        self.setWindowTitle('自动跟随智能购物车 CQUT')  # 设置窗口标题
        self.setWindowIcon(QIcon('Icon/cqutLog.jpg'))  # 设置窗口图标
