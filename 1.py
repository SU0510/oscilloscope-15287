import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
import numpy as np

class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.timer = QTimer(self)
        self.x = 0

        self.timer.timeout.connect(self.update_waveform)
        self.timer.start(10)  # 每10毫秒更新一次波形图

    def update_waveform(self):
        self.x += 1
        self.label.update()

    def paintEvent(self, event):
        painter = QPainter(self.label)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(Qt.blue))

        width = self.label.width()
        height = self.label.height()
        y = height // 2

        amplitude = height // 3
        frequency = 50
        phase = 0

        for i in range(width):
            x = self.x + i
            value = amplitude * np.sin((2 * np.pi * frequency * x) / width + phase) + y
            painter.drawPoint(i, value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = WaveformWidget()
    widget.show()
    sys.exit(app.exec_())
