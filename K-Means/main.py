# -*- coding: utf-8 -*-
# __Author__: Sdite
# __Email__ : a122691411@gmail.com

import sys
import pandas as pd
from kmeans import kMeans, CLASSIFICATION, BEGINING, NEWCENTROID
from numpy import mat
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtCore import QTimer, pyqtSignal, QThread


# 用于在PyQt5显示matplotlib的图
class KMeansCanvas(FigureCanvas):
    # '''FigureCanvas的最终父类是QWidget'''

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.mark = ['dr', '*b', 'sg', 'pk', '^r', '+b', '<g', 'hk']  # 聚类的图标加颜色, 最多设置8个聚类

        # 新建一个绘图对象
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.suptitle("KMeans")                                   # 设置figure的名字

        # 建立一个子图。如果要建立复合图, 可以在这里修改
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # 定义FigureCanvas的尺寸策略, 使之尽可能的向外填充空间
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def kMeansPlot(self, dataSet, k, statusBar, button):
        self.kMeans = kMeans(dataSet, k)                                    # KMeans生成器

        # 设置Timer
        self.timerStart = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(
            lambda: self.update_figure(dataSet, k, statusBar, button))      # 每隔一段时间就会触发一次 update函数

        self.update_figure(dataSet, k, statusBar, button)                   # 第一次自己来调用, 后面交给定时器Timer

    # 打开文件后显示图
    def plotInit(self, dataSet, statusBar):
        self.axes.clear()  # 清除上次绘图结果
        for i in range(len(dataSet)):
            self.axes.plot(dataSet[i, 0], dataSet[i, 1], 'oc', markersize=5)
        self.draw()
        statusBar.showMessage('数据已显示...')


    def update_figure(self, dataSet, k, statusBar, button):
        self.axes.clear()  # 清除上次绘图结果
        try:
            centroids, clusterAssessment, status = next(self.kMeans)  # 用生成器来迭代
            centroidsX = centroids.getA()[:, 0]
            centroidsY = centroids.getA()[:, 1]

            if status == CLASSIFICATION:
                # 表明分了类
                statusBar.showMessage('正在分类...')
                for i in range(len(dataSet)):
                    self.axes.plot(dataSet[i, 0], dataSet[i, 1], self.mark[int(clusterAssessment[i, 0])], markersize=5)

            elif status == NEWCENTROID:
                # 生成了新的聚类中心
                statusBar.showMessage('更新聚类中心...')
                for i in range(len(dataSet)):
                    self.axes.plot(dataSet[i, 0], dataSet[i, 1], self.mark[int(clusterAssessment[i, 0])], markersize=5)

            elif status == BEGINING:
                statusBar.showMessage('生成随机聚类中心...')
                # 最开始状态没有分类, 所以未分类的颜色点颜色都是一样的
                for i in range(len(dataSet)):
                    self.axes.plot(dataSet[i, 0], dataSet[i, 1], 'oc', markersize=5)

            legends = []
            for i in range(k):
                legend, = self.axes.plot(centroidsX[i], centroidsY[i], self.mark[i], markersize=10)
                legends.append(legend)
            self.axes.legend(legends, list(range(k)), loc="upper left")

            self.draw()

            if self.timerStart:
                self.timer.start(2000)
                self.timerStart = False
        except:
            statusBar.showMessage('KMeans已收敛')
            button.setEnabled(True)
            self.timer.stop()                       # 聚类收敛了, 停止迭代器


class OpenFileThread(QThread):
    endTrigger = pyqtSignal()

    def __init__(self, target, args):
        super(OpenFileThread, self).__init__()
        self.fun = target
        self.args = args

    def run(self):
        self.fun(*self.args)
        self.endTrigger.emit()


class KMeansApp(QMainWindow):

    def __init__(self):
        super(KMeansApp, self).__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('KMeans')
        self.setWindowIcon(QIcon('Images/K.jpg'))
        self.resize(900, 800)

        self.mainWidget = QWidget()                     # 主窗体控件
        self.mainLayout = QVBoxLayout()                 # 主窗体layout
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

        self.menuBar()                                  # 菜单栏
        self.statusBar()                                # 状态栏
        self.setOpenFileMenu()                          # 打开文件菜单
        self.initBtn()                                  # 初始化按钮
        self.setMatplotlibWiget()                       # 初始化matplotlib的控件

        self.center()
        self.show()

    # 设置打开文件的功能
    def setOpenFileMenu(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('File')

        openFile = QAction(QIcon('Images/open.png'), 'open file', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('打开文件')
        openFile.triggered.connect(self.showFileDialog)

        fileMenu.addAction(openFile)

    # 显示文件对话框
    def showFileDialog(self):
        # 打开文件,只允许打开'csv'文件
        filePath = QFileDialog.getOpenFileName(self, caption='打开文件', directory='./',
                                               filter='*.csv')
        if filePath[0]:
            file = filePath[0]
            self.t = OpenFileThread(target=KMeansApp.openFile, args=(self, file))
            self.t.endTrigger.connect(lambda : self.openFileSignnal(file))
            self.t.start()

    def openFileSignnal(self, file):
        self.mpl.plotInit(self.dataSet, self.statusBar())
        self.kButton.setEnabled(True)
        self.setWindowTitle('KMeans - ' + file)

    # 打开文件
    def openFile(self, file):
        self.dataSet = mat(pd.read_csv(file, dtype='float64')) # 数据只取前两列

    def initBtn(self):
        # 聚类个数输入框
        kLabel = QLabel()
        kLabel.setText("k: ")
        self.kEdit = QLineEdit()
        kIntValidator = QIntValidator(self)
        kIntValidator.setRange(1, 8)
        self.kEdit.setPlaceholderText("分类个数(范围: 1-8)")
        self.kEdit.setValidator(kIntValidator)

        self.kButton = QPushButton()
        self.kButton.setText("分类")
        self.kButton.setEnabled(False)
        self.kButton.clicked.connect(self.kMeansPlot)

        hBox = QHBoxLayout(self)
        hBox.addWidget(kLabel, 0)
        hBox.addWidget(self.kEdit, 0)
        hBox.addWidget(self.kButton, 0)

        hWidget = QWidget()
        hWidget.setLayout(hBox)
        self.mainLayout.addWidget(hWidget)

    def setMatplotlibWiget(self):
        self.mpl = KMeansCanvas(self)
        self.mplNtb = NavigationToolbar(self.mpl, self)  # matplotlib添加完整的工具栏
        self.mainLayout.addWidget(self.mpl)
        self.mainLayout.addWidget(self.mplNtb)

    def kMeansPlot(self):
        if self.kEdit.text() == "":
            QMessageBox.warning(self, "警告", "请输入分类个数", QMessageBox.Ok)
            return

        self.kButton.setEnabled(False)

        # 替换掉旧的Canvas
        self.newMpl = KMeansCanvas(self)
        self.newNplNtb = NavigationToolbar(self.newMpl, self)
        self.mainLayout.replaceWidget(self.mpl, self.newMpl)  # 替换控件
        self.mainLayout.replaceWidget(self.newNplNtb, self.mplNtb)
        del self.mpl, self.mplNtb
        self.mpl = self.newMpl
        self.mplNtb = self.newNplNtb

        # 开始画KMeans结果
        self.mpl.kMeansPlot(self.dataSet, int(self.kEdit.text()), self.statusBar(), self.kButton)

    # 窗口居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    executor = KMeansApp()
    sys.exit(app.exec_())
