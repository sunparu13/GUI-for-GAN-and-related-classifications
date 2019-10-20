# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'total_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1440, 798)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.batch_gan = QtWidgets.QLabel(self.centralwidget)
        self.batch_gan.setGeometry(QtCore.QRect(300, 90, 111, 20))
        self.batch_gan.setObjectName("batch_gan")
        self.cl_epochs_button = QtWidgets.QPushButton(self.centralwidget)
        self.cl_epochs_button.setGeometry(QtCore.QRect(180, 720, 141, 21))
        self.cl_epochs_button.setObjectName("cl_epochs_button")
        self.whole_title = QtWidgets.QLabel(self.centralwidget)
        self.whole_title.setGeometry(QtCore.QRect(20, -10, 651, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(24)
        self.whole_title.setFont(font)
        self.whole_title.setObjectName("whole_title")
        self.optimizer_d = QtWidgets.QLabel(self.centralwidget)
        self.optimizer_d.setGeometry(QtCore.QRect(580, 90, 111, 20))
        self.optimizer_d.setObjectName("optimizer_d")
        self.epochs_gan = QtWidgets.QLabel(self.centralwidget)
        self.epochs_gan.setGeometry(QtCore.QRect(420, 90, 131, 20))
        self.epochs_gan.setObjectName("epochs_gan")
        self.acc = QtWidgets.QLabel(self.centralwidget)
        self.acc.setGeometry(QtCore.QRect(640, 600, 81, 16))
        self.acc.setObjectName("acc")
        self.gan_display_box = QtWidgets.QLabel(self.centralwidget)
        self.gan_display_box.setGeometry(QtCore.QRect(920, 50, 480, 480))
        self.gan_display_box.setFrameShape(QtWidgets.QFrame.Box)
        self.gan_display_box.setFrameShadow(QtWidgets.QFrame.Plain)
        self.gan_display_box.setText("")
        self.gan_display_box.setObjectName("gan_display_box")
        self.cm = QtWidgets.QLabel(self.centralwidget)
        self.cm.setGeometry(QtCore.QRect(920, 590, 81, 16))
        self.cm.setObjectName("cm")
        self.metrics = QtWidgets.QLabel(self.centralwidget)
        self.metrics.setGeometry(QtCore.QRect(1120, 550, 171, 16))
        self.metrics.setObjectName("metrics")
        self.cm_box = QtWidgets.QLabel(self.centralwidget)
        self.cm_box.setGeometry(QtCore.QRect(920, 620, 141, 131))
        self.cm_box.setFrameShape(QtWidgets.QFrame.Box)
        self.cm_box.setFrameShadow(QtWidgets.QFrame.Plain)
        self.cm_box.setText("")
        self.cm_box.setObjectName("cm_box")
        self.loss_image_box = QtWidgets.QLabel(self.centralwidget)
        self.loss_image_box.setGeometry(QtCore.QRect(530, 470, 361, 121))
        self.loss_image_box.setFrameShape(QtWidgets.QFrame.Box)
        self.loss_image_box.setFrameShadow(QtWidgets.QFrame.Plain)
        self.loss_image_box.setText("")
        self.loss_image_box.setObjectName("loss_image_box")
        self.gan_save_box = QtWidgets.QTextEdit(self.centralwidget)
        self.gan_save_box.setGeometry(QtCore.QRect(480, 400, 411, 41))
        self.gan_save_box.setObjectName("gan_save_box")
        self.gen = QtWidgets.QLabel(self.centralwidget)
        self.gen.setGeometry(QtCore.QRect(30, 90, 81, 20))
        self.gen.setObjectName("gen")
        self.GAN_title = QtWidgets.QLabel(self.centralwidget)
        self.GAN_title.setGeometry(QtCore.QRect(20, 50, 341, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.GAN_title.setFont(font)
        self.GAN_title.setObjectName("GAN_title")
        self.dis = QtWidgets.QLabel(self.centralwidget)
        self.dis.setGeometry(QtCore.QRect(170, 90, 61, 21))
        self.dis.setObjectName("dis")
        self.k_d = QtWidgets.QLabel(self.centralwidget)
        self.k_d.setGeometry(QtCore.QRect(20, 160, 231, 20))
        self.k_d.setObjectName("k_d")
        self.k_g_box_2 = QtWidgets.QLabel(self.centralwidget)
        self.k_g_box_2.setGeometry(QtCore.QRect(270, 160, 231, 20))
        self.k_g_box_2.setObjectName("k_g_box_2")
        self.optimizer_g = QtWidgets.QLabel(self.centralwidget)
        self.optimizer_g.setGeometry(QtCore.QRect(710, 90, 111, 20))
        self.optimizer_g.setObjectName("optimizer_g")
        self.channel = QtWidgets.QLabel(self.centralwidget)
        self.channel.setGeometry(QtCore.QRect(510, 160, 161, 20))
        self.channel.setObjectName("channel")
        self.z_dim = QtWidgets.QLabel(self.centralwidget)
        self.z_dim.setGeometry(QtCore.QRect(690, 160, 231, 20))
        self.z_dim.setObjectName("z_dim")
        self.gan_save = QtWidgets.QLabel(self.centralwidget)
        self.gan_save.setGeometry(QtCore.QRect(480, 380, 411, 20))
        self.gan_save.setObjectName("gan_save")
        self.gan_display_2 = QtWidgets.QLabel(self.centralwidget)
        self.gan_display_2.setGeometry(QtCore.QRect(980, 10, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gan_display_2.setFont(font)
        self.gan_display_2.setObjectName("gan_display_2")
        self.dcgan_loops_button = QtWidgets.QPushButton(self.centralwidget)
        self.dcgan_loops_button.setGeometry(QtCore.QRect(230, 361, 201, 21))
        self.dcgan_loops_button.setObjectName("dcgan_loops_button")
        self.dcgan_label = QtWidgets.QLabel(self.centralwidget)
        self.dcgan_label.setGeometry(QtCore.QRect(20, 350, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dcgan_label.setFont(font)
        self.dcgan_label.setObjectName("dcgan_label")
        self.wgan_div_label = QtWidgets.QLabel(self.centralwidget)
        self.wgan_div_label.setGeometry(QtCore.QRect(20, 400, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.wgan_div_label.setFont(font)
        self.wgan_div_label.setObjectName("wgan_div_label")
        self.dcgan_epochs_button = QtWidgets.QPushButton(self.centralwidget)
        self.dcgan_epochs_button.setGeometry(QtCore.QRect(230, 381, 201, 21))
        self.dcgan_epochs_button.setObjectName("dcgan_epochs_button")
        self.wgan_epochs_button = QtWidgets.QPushButton(self.centralwidget)
        self.wgan_epochs_button.setGeometry(QtCore.QRect(230, 431, 201, 21))
        self.wgan_epochs_button.setObjectName("wgan_epochs_button")
        self.wgan_loops_button = QtWidgets.QPushButton(self.centralwidget)
        self.wgan_loops_button.setGeometry(QtCore.QRect(230, 410, 201, 21))
        self.wgan_loops_button.setObjectName("wgan_loops_button")
        self.image_6 = QtWidgets.QLabel(self.centralwidget)
        self.image_6.setGeometry(QtCore.QRect(20, 230, 401, 111))
        self.image_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.image_6.setFrameShape(QtWidgets.QFrame.Box)
        self.image_6.setFrameShadow(QtWidgets.QFrame.Plain)
        self.image_6.setObjectName("image_6")
        self.smooth_real = QtWidgets.QLabel(self.centralwidget)
        self.smooth_real.setGeometry(QtCore.QRect(180, 240, 81, 20))
        self.smooth_real.setObjectName("smooth_real")
        self.smooth_fake = QtWidgets.QLabel(self.centralwidget)
        self.smooth_fake.setGeometry(QtCore.QRect(300, 240, 81, 20))
        self.smooth_fake.setObjectName("smooth_fake")
        self.gan_npy_box = QtWidgets.QTextEdit(self.centralwidget)
        self.gan_npy_box.setGeometry(QtCore.QRect(480, 329, 411, 41))
        self.gan_npy_box.setObjectName("gan_npy_box")
        self.gan_npy = QtWidgets.QLabel(self.centralwidget)
        self.gan_npy.setGeometry(QtCore.QRect(500, 300, 191, 20))
        self.gan_npy.setObjectName("gan_npy")
        self.Classification_title = QtWidgets.QLabel(self.centralwidget)
        self.Classification_title.setGeometry(QtCore.QRect(20, 510, 481, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.Classification_title.setFont(font)
        self.Classification_title.setObjectName("Classification_title")
        self.GAN_model_box = QtWidgets.QTextEdit(self.centralwidget)
        self.GAN_model_box.setGeometry(QtCore.QRect(10, 560, 231, 41))
        self.GAN_model_box.setObjectName("GAN_model_box")
        self.GAN_classification_npy = QtWidgets.QLabel(self.centralwidget)
        self.GAN_classification_npy.setGeometry(QtCore.QRect(20, 529, 321, 31))
        self.GAN_classification_npy.setObjectName("GAN_classification_npy")
        self.optimizer_cl = QtWidgets.QLabel(self.centralwidget)
        self.optimizer_cl.setGeometry(QtCore.QRect(30, 620, 111, 20))
        self.optimizer_cl.setObjectName("optimizer_cl")
        self.epochs_cl = QtWidgets.QLabel(self.centralwidget)
        self.epochs_cl.setGeometry(QtCore.QRect(240, 630, 81, 20))
        self.epochs_cl.setObjectName("epochs_cl")
        self.loss = QtWidgets.QLabel(self.centralwidget)
        self.loss.setGeometry(QtCore.QRect(640, 450, 141, 16))
        self.loss.setObjectName("loss")
        self.acc_image_box = QtWidgets.QLabel(self.centralwidget)
        self.acc_image_box.setGeometry(QtCore.QRect(530, 630, 361, 121))
        self.acc_image_box.setFrameShape(QtWidgets.QFrame.Box)
        self.acc_image_box.setFrameShadow(QtWidgets.QFrame.Plain)
        self.acc_image_box.setText("")
        self.acc_image_box.setObjectName("acc_image_box")
        self.metrics_box = QtWidgets.QLabel(self.centralwidget)
        self.metrics_box.setGeometry(QtCore.QRect(1090, 580, 321, 171))
        self.metrics_box.setFrameShape(QtWidgets.QFrame.Box)
        self.metrics_box.setFrameShadow(QtWidgets.QFrame.Plain)
        self.metrics_box.setText("")
        self.metrics_box.setObjectName("metrics_box")
        self.result_saving_cl = QtWidgets.QLabel(self.centralwidget)
        self.result_saving_cl.setGeometry(QtCore.QRect(20, 690, 421, 20))
        self.result_saving_cl.setObjectName("result_saving_cl")
        self.result_saving_cl_box = QtWidgets.QTextEdit(self.centralwidget)
        self.result_saving_cl_box.setGeometry(QtCore.QRect(20, 720, 141, 31))
        self.result_saving_cl_box.setObjectName("result_saving_cl_box")
        self.dcgan_labe_explain = QtWidgets.QLabel(self.centralwidget)
        self.dcgan_labe_explain.setGeometry(QtCore.QRect(40, 300, 351, 20))
        self.dcgan_labe_explain.setObjectName("dcgan_labe_explain")
        self.show_images_button = QtWidgets.QPushButton(self.centralwidget)
        self.show_images_button.setGeometry(QtCore.QRect(1230, 10, 113, 32))
        self.show_images_button.setObjectName("show_images_button")
        self.k_g_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.k_g_spinBox.setGeometry(QtCore.QRect(320, 180, 121, 41))
        self.k_g_spinBox.setObjectName("k_g_spinBox")
        self.smooth_real_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.smooth_real_doubleSpinBox.setGeometry(QtCore.QRect(180, 260, 68, 24))
        self.smooth_real_doubleSpinBox.setObjectName("smooth_real_doubleSpinBox")
        self.k_d_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.k_d_spinBox.setGeometry(QtCore.QRect(90, 180, 121, 41))
        self.k_d_spinBox.setObjectName("k_d_spinBox")
        self.epochs_gan_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.epochs_gan_spinBox.setGeometry(QtCore.QRect(430, 110, 121, 41))
        self.epochs_gan_spinBox.setObjectName("epochs_gan_spinBox")
        self.bs_gan_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.bs_gan_spinBox.setGeometry(QtCore.QRect(290, 110, 121, 41))
        self.bs_gan_spinBox.setObjectName("bs_gan_spinBox")
        self.channel_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.channel_spinBox.setGeometry(QtCore.QRect(530, 180, 121, 41))
        self.channel_spinBox.setObjectName("channel_spinBox")
        self.z_dim_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.z_dim_spinBox.setGeometry(QtCore.QRect(750, 180, 121, 41))
        self.z_dim_spinBox.setObjectName("z_dim_spinBox")
        self.smooth_fake_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.smooth_fake_doubleSpinBox.setGeometry(QtCore.QRect(300, 260, 68, 24))
        self.smooth_fake_doubleSpinBox.setObjectName("smooth_fake_doubleSpinBox")
        self.epochs_cl_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.epochs_cl_spinBox.setGeometry(QtCore.QRect(330, 620, 121, 31))
        self.epochs_cl_spinBox.setObjectName("epochs_cl_spinBox")
        self.train_npy_button = QtWidgets.QPushButton(self.centralwidget)
        self.train_npy_button.setGeometry(QtCore.QRect(690, 300, 101, 21))
        self.train_npy_button.setObjectName("train_npy_button")
        self.gen_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.gen_comboBox.setGeometry(QtCore.QRect(20, 110, 101, 41))
        self.gen_comboBox.setObjectName("gen_comboBox")
        self.gen_comboBox.addItem("")
        self.gen_comboBox.setItemText(0, "")
        self.gen_comboBox.addItem("")
        self.gen_comboBox.addItem("")
        self.gen_comboBox.addItem("")
        self.gen_comboBox.addItem("")
        self.gen_comboBox.addItem("")
        self.dis_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.dis_comboBox.setGeometry(QtCore.QRect(150, 110, 101, 41))
        self.dis_comboBox.setObjectName("dis_comboBox")
        self.dis_comboBox.addItem("")
        self.dis_comboBox.setItemText(0, "")
        self.dis_comboBox.addItem("")
        self.dis_comboBox.addItem("")
        self.dis_comboBox.addItem("")
        self.dis_comboBox.addItem("")
        self.dis_comboBox.addItem("")
        self.dis_comboBox.addItem("")
        self.interval_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.interval_spinBox.setGeometry(QtCore.QRect(530, 240, 121, 41))
        self.interval_spinBox.setObjectName("interval_spinBox")
        self.channel_2 = QtWidgets.QLabel(self.centralwidget)
        self.channel_2.setGeometry(QtCore.QRect(540, 220, 91, 20))
        self.channel_2.setObjectName("channel_2")
        self.optimizer_d_lr_spinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.optimizer_d_lr_spinBox.setGeometry(QtCore.QRect(570, 110, 121, 24))
        self.optimizer_d_lr_spinBox.setDecimals(5)
        self.optimizer_d_lr_spinBox.setObjectName("optimizer_d_lr_spinBox")
        self.optimizer_d_b_spinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.optimizer_d_b_spinBox.setGeometry(QtCore.QRect(570, 130, 121, 24))
        self.optimizer_d_b_spinBox.setObjectName("optimizer_d_b_spinBox")
        self.optimizer_g_b_spinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.optimizer_g_b_spinBox.setGeometry(QtCore.QRect(710, 130, 121, 24))
        self.optimizer_g_b_spinBox.setObjectName("optimizer_g_b_spinBox")
        self.optimizer_g_lr_spinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.optimizer_g_lr_spinBox.setGeometry(QtCore.QRect(710, 110, 121, 24))
        self.optimizer_g_lr_spinBox.setDecimals(5)
        self.optimizer_g_lr_spinBox.setObjectName("optimizer_g_lr_spinBox")
        self.display_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.display_spinBox.setGeometry(QtCore.QRect(750, 240, 121, 41))
        self.display_spinBox.setObjectName("display_spinBox")
        self.channel_3 = QtWidgets.QLabel(self.centralwidget)
        self.channel_3.setGeometry(QtCore.QRect(750, 220, 121, 20))
        self.channel_3.setObjectName("channel_3")
        self.select_model_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_model_button.setGeometry(QtCore.QRect(260, 570, 71, 31))
        self.select_model_button.setObjectName("select_model_button")
        self.optimizer_cl_b_spinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.optimizer_cl_b_spinBox.setGeometry(QtCore.QRect(30, 660, 121, 24))
        self.optimizer_cl_b_spinBox.setObjectName("optimizer_cl_b_spinBox")
        self.optimizer_cl_lr_spinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.optimizer_cl_lr_spinBox.setGeometry(QtCore.QRect(30, 640, 121, 24))
        self.optimizer_cl_lr_spinBox.setDecimals(5)
        self.optimizer_cl_lr_spinBox.setObjectName("optimizer_cl_lr_spinBox")
        self.cl_pitting_amount_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.cl_pitting_amount_spinBox.setGeometry(QtCore.QRect(380, 560, 121, 41))
        self.cl_pitting_amount_spinBox.setObjectName("cl_pitting_amount_spinBox")
        self.dis_2 = QtWidgets.QLabel(self.centralwidget)
        self.dis_2.setGeometry(QtCore.QRect(330, 529, 181, 31))
        self.dis_2.setObjectName("dis_2")
        self.select_loss_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_loss_button.setGeometry(QtCore.QRect(740, 450, 71, 20))
        self.select_loss_button.setObjectName("select_loss_button")
        self.select_acc_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_acc_button.setGeometry(QtCore.QRect(740, 600, 71, 21))
        self.select_acc_button.setObjectName("select_acc_button")
        self.select_cm_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_cm_button.setGeometry(QtCore.QRect(1010, 590, 51, 21))
        self.select_cm_button.setObjectName("select_cm_button")
        self.select_table_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_table_button.setGeometry(QtCore.QRect(1320, 550, 71, 21))
        self.select_table_button.setObjectName("select_table_button")
        self.bs_cl_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.bs_cl_spinBox.setGeometry(QtCore.QRect(330, 660, 121, 31))
        self.bs_cl_spinBox.setObjectName("bs_cl_spinBox")
        self.bs_cl = QtWidgets.QLabel(self.centralwidget)
        self.bs_cl.setGeometry(QtCore.QRect(200, 660, 121, 31))
        self.bs_cl.setObjectName("bs_cl")
        self.lsgan_epochs_button = QtWidgets.QPushButton(self.centralwidget)
        self.lsgan_epochs_button.setGeometry(QtCore.QRect(230, 481, 201, 21))
        self.lsgan_epochs_button.setObjectName("lsgan_epochs_button")
        self.lsgan_loops_button = QtWidgets.QPushButton(self.centralwidget)
        self.lsgan_loops_button.setGeometry(QtCore.QRect(230, 460, 201, 21))
        self.lsgan_loops_button.setObjectName("lsgan_loops_button")
        self.lsgan_label = QtWidgets.QLabel(self.centralwidget)
        self.lsgan_label.setGeometry(QtCore.QRect(20, 450, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lsgan_label.setFont(font)
        self.lsgan_label.setObjectName("lsgan_label")
        self.image_6.raise_()
        self.batch_gan.raise_()
        self.cl_epochs_button.raise_()
        self.whole_title.raise_()
        self.optimizer_d.raise_()
        self.epochs_gan.raise_()
        self.acc.raise_()
        self.gan_display_box.raise_()
        self.cm.raise_()
        self.metrics.raise_()
        self.cm_box.raise_()
        self.loss_image_box.raise_()
        self.gan_save_box.raise_()
        self.gen.raise_()
        self.GAN_title.raise_()
        self.dis.raise_()
        self.k_d.raise_()
        self.k_g_box_2.raise_()
        self.optimizer_g.raise_()
        self.channel.raise_()
        self.z_dim.raise_()
        self.gan_save.raise_()
        self.gan_display_2.raise_()
        self.dcgan_loops_button.raise_()
        self.dcgan_label.raise_()
        self.wgan_div_label.raise_()
        self.dcgan_epochs_button.raise_()
        self.wgan_epochs_button.raise_()
        self.wgan_loops_button.raise_()
        self.smooth_real.raise_()
        self.smooth_fake.raise_()
        self.gan_npy_box.raise_()
        self.gan_npy.raise_()
        self.Classification_title.raise_()
        self.GAN_model_box.raise_()
        self.GAN_classification_npy.raise_()
        self.optimizer_cl.raise_()
        self.epochs_cl.raise_()
        self.loss.raise_()
        self.acc_image_box.raise_()
        self.metrics_box.raise_()
        self.result_saving_cl.raise_()
        self.result_saving_cl_box.raise_()
        self.dcgan_labe_explain.raise_()
        self.show_images_button.raise_()
        self.k_g_spinBox.raise_()
        self.smooth_real_doubleSpinBox.raise_()
        self.k_d_spinBox.raise_()
        self.epochs_gan_spinBox.raise_()
        self.bs_gan_spinBox.raise_()
        self.channel_spinBox.raise_()
        self.z_dim_spinBox.raise_()
        self.smooth_fake_doubleSpinBox.raise_()
        self.epochs_cl_spinBox.raise_()
        self.train_npy_button.raise_()
        self.gen_comboBox.raise_()
        self.dis_comboBox.raise_()
        self.interval_spinBox.raise_()
        self.channel_2.raise_()
        self.optimizer_d_lr_spinBox.raise_()
        self.optimizer_d_b_spinBox.raise_()
        self.optimizer_g_b_spinBox.raise_()
        self.optimizer_g_lr_spinBox.raise_()
        self.display_spinBox.raise_()
        self.channel_3.raise_()
        self.select_model_button.raise_()
        self.optimizer_cl_b_spinBox.raise_()
        self.optimizer_cl_lr_spinBox.raise_()
        self.cl_pitting_amount_spinBox.raise_()
        self.dis_2.raise_()
        self.select_loss_button.raise_()
        self.select_acc_button.raise_()
        self.select_cm_button.raise_()
        self.select_table_button.raise_()
        self.bs_cl_spinBox.raise_()
        self.bs_cl.raise_()
        self.lsgan_epochs_button.raise_()
        self.lsgan_loops_button.raise_()
        self.lsgan_label.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1440, 21))
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
        self.batch_gan.setText(_translate("MainWindow", "batch size_GAN"))
        self.cl_epochs_button.setText(_translate("MainWindow", "train and test"))
        self.whole_title.setText(_translate("MainWindow", "Interface for GAN and its related classification task"))
        self.optimizer_d.setText(_translate("MainWindow", "optimizer D adam"))
        self.epochs_gan.setText(_translate("MainWindow", "training loops/epochs"))
        self.acc.setText(_translate("MainWindow", "Accuracy curve"))
        self.cm.setText(_translate("MainWindow", "Confusion Matrix"))
        self.metrics.setText(_translate("MainWindow", "Metrics Table, 0-pitting, 1-intact"))
        self.gen.setText(_translate("MainWindow", "Generator"))
        self.GAN_title.setText(_translate("MainWindow", "Input parameters for GAN training"))
        self.dis.setText(_translate("MainWindow", "Discriminator"))
        self.k_d.setText(_translate("MainWindow", "Training round D in one loop/epoch"))
        self.k_g_box_2.setText(_translate("MainWindow", "Training round G in one loop/epoch"))
        self.optimizer_g.setText(_translate("MainWindow", "optimizer G adam"))
        self.channel.setText(_translate("MainWindow", "Training images channels"))
        self.z_dim.setText(_translate("MainWindow", "latent vector size for generator input"))
        self.gan_save.setText(_translate("MainWindow", "result saving folder name (incl. display images, generated images, model)"))
        self.gan_display_2.setText(_translate("MainWindow", "Image generation result display"))
        self.dcgan_loops_button.setText(_translate("MainWindow", "Train with loop-wise"))
        self.dcgan_label.setText(_translate("MainWindow", "Training with DCGAN"))
        self.wgan_div_label.setText(_translate("MainWindow", "Training with WGAN-div"))
        self.dcgan_epochs_button.setText(_translate("MainWindow", "Train with epoch-wise"))
        self.wgan_epochs_button.setText(_translate("MainWindow", "Train with epoch-wise"))
        self.wgan_loops_button.setText(_translate("MainWindow", "Train with loop-wise"))
        self.image_6.setText(_translate("MainWindow", "special for DCGAN"))
        self.smooth_real.setText(_translate("MainWindow", "real label"))
        self.smooth_fake.setText(_translate("MainWindow", "fake label"))
        self.gan_npy.setText(_translate("MainWindow", "training image dataset (load npy file)"))
        self.Classification_title.setText(_translate("MainWindow", "Input parameters for Classification (replace real images)"))
        self.GAN_classification_npy.setText(_translate("MainWindow", "select GAN model to generate synthetic images "))
        self.optimizer_cl.setText(_translate("MainWindow", "optimizer adam"))
        self.epochs_cl.setText(_translate("MainWindow", "training epochs"))
        self.loss.setText(_translate("MainWindow", "Loss function curve"))
        self.result_saving_cl.setText(_translate("MainWindow", "result saving folder name (incl. loss and acc values, confusion metrics, metrics table))"))
        self.dcgan_labe_explain.setText(_translate("MainWindow", "original: real-1, fake-0. when smoothed, this can be tuned"))
        self.show_images_button.setText(_translate("MainWindow", "select images"))
        self.train_npy_button.setText(_translate("MainWindow", "select"))
        self.gen_comboBox.setItemText(1, _translate("MainWindow", "gen1"))
        self.gen_comboBox.setItemText(2, _translate("MainWindow", "gen2"))
        self.gen_comboBox.setItemText(3, _translate("MainWindow", "gen3"))
        self.gen_comboBox.setItemText(4, _translate("MainWindow", "gen4"))
        self.gen_comboBox.setItemText(5, _translate("MainWindow", "gen5"))
        self.dis_comboBox.setItemText(1, _translate("MainWindow", "dis1"))
        self.dis_comboBox.setItemText(2, _translate("MainWindow", "dis2"))
        self.dis_comboBox.setItemText(3, _translate("MainWindow", "dis3"))
        self.dis_comboBox.setItemText(4, _translate("MainWindow", "dis4"))
        self.dis_comboBox.setItemText(5, _translate("MainWindow", "dis5"))
        self.dis_comboBox.setItemText(6, _translate("MainWindow", "dis6"))
        self.channel_2.setText(_translate("MainWindow", "saving interval"))
        self.channel_3.setText(_translate("MainWindow", "display row/column"))
        self.select_model_button.setText(_translate("MainWindow", "select"))
        self.dis_2.setText(_translate("MainWindow", "generated images amount, best 700"))
        self.select_loss_button.setText(_translate("MainWindow", "select"))
        self.select_acc_button.setText(_translate("MainWindow", "select"))
        self.select_cm_button.setText(_translate("MainWindow", "select"))
        self.select_table_button.setText(_translate("MainWindow", "select"))
        self.bs_cl.setText(_translate("MainWindow", "batch_size_classification"))
        self.lsgan_epochs_button.setText(_translate("MainWindow", "Train with epoch-wise"))
        self.lsgan_loops_button.setText(_translate("MainWindow", "Train with loop-wise"))
        self.lsgan_label.setText(_translate("MainWindow", "Training with LSGAN"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

