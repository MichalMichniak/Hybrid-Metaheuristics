# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitled.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QTextBrowser, QWidget)
from constants import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(500, 80, 93, 28))
        self.N_numer = QLineEdit(self.centralwidget)
        self.N_numer.setObjectName(u"N_numer")
        self.N_numer.setGeometry(QRect(170, 40, 113, 22))
        self.N = QLabel(self.centralwidget)
        self.N.setObjectName(u"N")
        self.N.setGeometry(QRect(54, 40, 71, 21))
        self.Taboo_neighbours = QLabel(self.centralwidget)
        self.Taboo_neighbours.setObjectName(u"Taboo_neighbours")
        self.Taboo_neighbours.setGeometry(QRect(30, 80, 111, 16))
        self.surv_part = QLabel(self.centralwidget)
        self.surv_part.setObjectName(u"surv_part")
        self.surv_part.setGeometry(QRect(30, 120, 55, 16))
        self.mut_prob = QLabel(self.centralwidget)
        self.mut_prob.setObjectName(u"mut_prob")
        self.mut_prob.setGeometry(QRect(30, 150, 55, 16))
        self.M_PSO = QLabel(self.centralwidget)
        self.M_PSO.setObjectName(u"M_PSO")
        self.M_PSO.setGeometry(QRect(30, 180, 55, 16))
        self.M_taboo = QLabel(self.centralwidget)
        self.M_taboo.setObjectName(u"M_taboo")
        self.M_taboo.setGeometry(QRect(30, 210, 55, 16))
        self.M_start = QLabel(self.centralwidget)
        self.M_start.setObjectName(u"M_start")
        self.M_start.setGeometry(QRect(30, 240, 55, 16))
        self.m_species = QLabel(self.centralwidget)
        self.m_species.setObjectName(u"m_species")
        self.m_species.setGeometry(QRect(30, 280, 55, 16))
        self.max_iter = QLabel(self.centralwidget)
        self.max_iter.setObjectName(u"max_iter")
        self.max_iter.setGeometry(QRect(30, 320, 55, 16))
        self.taboo_neighbours_num = QLineEdit(self.centralwidget)
        self.taboo_neighbours_num.setObjectName(u"taboo_neighbours_num")
        self.taboo_neighbours_num.setGeometry(QRect(170, 80, 113, 22))
        self.surr_part_num = QLineEdit(self.centralwidget)
        self.surr_part_num.setObjectName(u"surr_part_num")
        self.surr_part_num.setGeometry(QRect(170, 120, 113, 22))
        self.mut_prob_num = QLineEdit(self.centralwidget)
        self.mut_prob_num.setObjectName(u"mut_prob_num")
        self.mut_prob_num.setGeometry(QRect(170, 150, 113, 22))
        self.m_pso_num = QLineEdit(self.centralwidget)
        self.m_pso_num.setObjectName(u"m_pso_num")
        self.m_pso_num.setGeometry(QRect(170, 180, 113, 22))
        self.m_taboo_num = QLineEdit(self.centralwidget)
        self.m_taboo_num.setObjectName(u"m_taboo_num")
        self.m_taboo_num.setGeometry(QRect(170, 210, 113, 22))
        self.m_start_num = QLineEdit(self.centralwidget)
        self.m_start_num.setObjectName(u"lineEdit")
        self.m_start_num.setGeometry(QRect(170, 240, 113, 22))
        self.m_species_num = QLineEdit(self.centralwidget)
        self.m_species_num.setObjectName(u"m_species_num")
        self.m_species_num.setGeometry(QRect(170, 270, 113, 22))
        self.max_iter_num = QLineEdit(self.centralwidget)
        self.max_iter_num.setObjectName(u"max_iter_num")
        self.max_iter_num.setGeometry(QRect(170, 320, 113, 22))
        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(370, 140, 341, 291))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"HYBRID METAHEURISTICS", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"START", None))
        self.N_numer.setText(QCoreApplication.translate("MainWindow", str(N), None))
        self.N.setText(QCoreApplication.translate("MainWindow", u"N ", None))
        self.Taboo_neighbours.setText(QCoreApplication.translate("MainWindow", u"Taboo_neighbours", None))
        self.surv_part.setText(QCoreApplication.translate("MainWindow", u"surv_part", None))
        self.mut_prob.setText(QCoreApplication.translate("MainWindow", u"mut_prob", None))
        self.M_PSO.setText(QCoreApplication.translate("MainWindow", u"M_PSO", None))
        self.M_taboo.setText(QCoreApplication.translate("MainWindow", u"M_Taboo", None))
        self.M_start.setText(QCoreApplication.translate("MainWindow", u"M_start", None))
        self.m_species.setText(QCoreApplication.translate("MainWindow", u"M_species", None))
        self.max_iter.setText(QCoreApplication.translate("MainWindow", u"max_iter", None))
        self.taboo_neighbours_num.setText(QCoreApplication.translate("MainWindow", str(TABOO_NEIGHBORS), None))
        self.surr_part_num.setText(QCoreApplication.translate("MainWindow", str(SURV_PART), None))
        self.mut_prob_num.setText(QCoreApplication.translate("MainWindow", str(MUTATION_PROB), None))
        self.m_pso_num.setText(QCoreApplication.translate("MainWindow", str(M_PSO), None))
        self.m_taboo_num.setText(QCoreApplication.translate("MainWindow", str(M_TABOO), None))
        self.m_start_num.setText(QCoreApplication.translate("MainWindow", str(START), None))
        self.m_species_num.setText(QCoreApplication.translate("MainWindow", str(M_SPECIES), None))
        self.max_iter_num.setText(QCoreApplication.translate("MainWindow", str(MAX_ITER), None))
    # retranslateUi

