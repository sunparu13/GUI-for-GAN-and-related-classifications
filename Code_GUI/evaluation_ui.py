# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluation_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 665)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.random_button = QtWidgets.QPushButton(self.centralwidget)
        self.random_button.setGeometry(QtCore.QRect(370, 120, 161, 31))
        self.random_button.setObjectName("random_button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 540, 81, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 160, 131, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(270, 120, 91, 24))
        self.spinBox.setObjectName("spinBox")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(50, 120, 211, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(50, 149, 211, 31))
        self.label_5.setObjectName("label_5")
        self.model_box = QtWidgets.QTextEdit(self.centralwidget)
        self.model_box.setGeometry(QtCore.QRect(170, 80, 251, 31))
        self.model_box.setObjectName("model_box")
        self.model_button = QtWidgets.QPushButton(self.centralwidget)
        self.model_button.setGeometry(QtCore.QRect(20, 80, 131, 31))
        self.model_button.setObjectName("model_button")
        self.display_box = QtWidgets.QLabel(self.centralwidget)
        self.display_box.setGeometry(QtCore.QRect(50, 180, 341, 341))
        self.display_box.setFrameShape(QtWidgets.QFrame.Box)
        self.display_box.setText("")
        self.display_box.setObjectName("display_box")
        self.fid_button = QtWidgets.QPushButton(self.centralwidget)
        self.fid_button.setGeometry(QtCore.QRect(50, 570, 101, 31))
        self.fid_button.setObjectName("fid_button")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(160, 560, 161, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.spinBox_tsne = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_tsne.setGeometry(QtCore.QRect(560, 540, 111, 21))
        self.spinBox_tsne.setObjectName("spinBox_tsne")
        self.tsne_button = QtWidgets.QPushButton(self.centralwidget)
        self.tsne_button.setGeometry(QtCore.QRect(680, 541, 71, 21))
        self.tsne_button.setObjectName("tsne_button")
        self.tsne_box = QtWidgets.QLabel(self.centralwidget)
        self.tsne_box.setGeometry(QtCore.QRect(410, 180, 341, 341))
        self.tsne_box.setFrameShape(QtWidgets.QFrame.Box)
        self.tsne_box.setText("")
        self.tsne_box.setObjectName("tsne_box")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(430, 540, 131, 16))
        self.label_6.setObjectName("label_6")
        self.z_dim_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.z_dim_spinBox.setGeometry(QtCore.QRect(550, 73, 121, 31))
        self.z_dim_spinBox.setObjectName("z_dim_spinBox")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(480, 20, 281, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(480, 40, 281, 31))
        self.label_8.setObjectName("label_8")
        self.original_box = QtWidgets.QTextEdit(self.centralwidget)
        self.original_box.setGeometry(QtCore.QRect(170, 50, 251, 31))
        self.original_box.setObjectName("original_box")
        self.original_button = QtWidgets.QPushButton(self.centralwidget)
        self.original_button.setGeometry(QtCore.QRect(20, 50, 131, 31))
        self.original_button.setObjectName("original_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.random_button.setText(_translate("MainWindow", "random generation"))
        self.label.setText(_translate("MainWindow", "FID SCORE"))
        self.label_2.setText(_translate("MainWindow", "t-SNE 2D display"))
        self.label_3.setText(_translate("MainWindow", "Evaluation"))
        self.label_4.setText(_translate("MainWindow", "rows&columns of generated display images "))
        self.label_5.setText(_translate("MainWindow", "generated display images"))
        self.model_button.setText(_translate("MainWindow", "choose generator model"))
        self.fid_button.setText(_translate("MainWindow", "calculate"))
        self.tsne_button.setText(_translate("MainWindow", "plot_tsne"))
        self.label_6.setText(_translate("MainWindow", "generated images amout"))
        self.label_7.setText(_translate("MainWindow", "latent variable dimension accroding to the trained model"))
        self.label_8.setText(_translate("MainWindow", "should be same as being tuned in GUI \"total_ui\""))
        self.original_button.setText(_translate("MainWindow", "choose original dataset"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

