from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTimeEdit
from PyQt5.QtCore import QTime
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "PyQt5 Window"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300

        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.MyTime()

        self.show()

    def MyTime(self):
        vbox = QVBoxLayout()
        time = QTime()
        time.setHMS(13, 15, 40)

        timeedit = QTimeEdit()
        timeedit.setFont(QtGui.QFont("Sanserif", 15))
        timeedit.setTime(time)

        vbox.addWidget(timeedit)

        self.setLayout(vbox)


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
