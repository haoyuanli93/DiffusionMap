# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\layout.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1116, 605)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(630, 40, 461, 511))
        self.tabWidget.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(16)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.general = QtWidgets.QWidget()
        self.general.setObjectName("general")
        self.tabWidget.addTab(self.general, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_3.setGeometry(QtCore.QRect(260, 30, 171, 151))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView.setGeometry(QtCore.QRect(50, 30, 161, 151))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_5 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_5.setGeometry(QtCore.QRect(50, 240, 161, 151))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_4.setGeometry(QtCore.QRect(260, 240, 171, 151))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.tabWidget.addTab(self.tab, "")
        self.roi = QtWidgets.QWidget()
        self.roi.setObjectName("roi")
        self.pushButton = QtWidgets.QPushButton(self.roi)
        self.pushButton.setGeometry(QtCore.QRect(10, 410, 261, 41))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.roi)
        self.label.setGeometry(QtCore.QRect(10, 200, 351, 31))
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(self.roi)
        self.textBrowser.setGeometry(QtCore.QRect(10, 230, 301, 41))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_2 = QtWidgets.QPushButton(self.roi)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 360, 261, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.tabWidget.addTab(self.roi, "")
        self.propogration = QtWidgets.QWidget()
        self.propogration.setObjectName("propogration")
        self.tabWidget.addTab(self.propogration, "")
        self.openGLWidget = QtWidgets.QOpenGLWidget(self.centralwidget)
        self.openGLWidget.setGeometry(QtCore.QRect(40, 50, 511, 501))
        self.openGLWidget.setObjectName("openGLWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1116, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.general), _translate("MainWindow", "General"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Inspect"))
        self.pushButton.setText(_translate("MainWindow", "Save Selected Region"))
        self.label.setText(_translate("MainWindow", "Number of Selected Samples"))
        self.pushButton_2.setText(_translate("MainWindow", "Select a Region"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.roi), _translate("MainWindow", "ROI"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.propogration), _translate("MainWindow", "Prop"))

