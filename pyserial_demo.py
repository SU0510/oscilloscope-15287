import sys
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer
from ui_demo_1 import Ui_Form
import sys
import multiprocessing
from PyQt5.QtGui import QGuiApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QCoreApplication
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import os

from PyQt5.QtGui import QPixmap
import numpy as np
from pyqtgraph import GraphicsLayoutWidget, PlotItem, AxisItem
data1 = [0] * 1000
data2 =[]
dataraw = []
class Pyqt5_Serial(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(Pyqt5_Serial, self).__init__()
        self.setupUi(self)
        self.init()
        self.setWindowTitle("串口示波器")
        self.ser = serial.Serial()
        self.port_check()
        self.num=[]
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        self.watch_2.setYRange(0, 0.1)





    def init(self):
        # 串口检测按钮
        self.s1__box_1.clicked.connect(self.port_check)
        self.savebut.clicked.connect(self.save)
        # 串口信息显示
        self.s1__box_2.currentTextChanged.connect(self.port_imf)
        self.choosefile.clicked.connect(self.open_file_dialog)
        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)
        self.playbut.clicked.connect(self.showdata)
        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)
        self.py_but.clicked.connect(self.runpy)
        # 发送数据按钮
        self.s3__send_button.clicked.connect(self.data_send)

        # 定时发送数据
        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer)

        self.kedu.valueChanged.connect(self.on_value_changed)
        # 定时器接收数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)

        # 清除发送窗口
        self.s3__clear_button.clicked.connect(self.send_data_clear)

        # 清除接收窗口
        self.s2__clear_button.clicked.connect(self.receive_data_clear)


    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.s1__box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")

    # 串口信息
    def port_imf(self):
        # 显示选定的串口的详细信息
        imf_s = self.s1__box_2.currentText()
        if imf_s != "":
            self.state_label.setText(self.Com_Dict[self.s1__box_2.currentText()])

    # 打开串口
    def port_open(self):
        self.ser.port = self.s1__box_2.currentText()
        self.ser.baudrate = int(self.s1__box_3.currentText())
        self.ser.bytesize = int(self.s1__box_4.currentText())
        self.ser.stopbits = int(self.s1__box_6.currentText())
        self.ser.parity = self.s1__box_5.currentText()

        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None

        # 打开串口接收定时器，周期为2ms
        self.timer.start(2)

        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
            self.formGroupBox1.setTitle("串口状态（已开启）")

    # 关闭串口
    def port_close(self):
        self.timer.stop()
        self.timer_send.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        self.formGroupBox1.setTitle("串口状态（已关闭）")

    # 发送数据
    def data_send(self):
        if self.ser.isOpen():
            input_s = self.s3__send_text.toPlainText()
            if input_s != "":
                # 非空字符串
                if self.hex_send.isChecked():
                    # hex发送
                    input_s = input_s.strip()
                    send_list = []
                    while input_s != '':
                        try:
                            num = int(input_s[0:2], 16)
                        except ValueError:
                            QMessageBox.critical(self, 'wrong data', '请输入十六进制数据，以空格分开!')
                            return None
                        input_s = input_s[2:].strip()
                        send_list.append(num)
                    input_s = bytes(send_list)
                else:
                    # ascii发送
                    input_s = (input_s + '\r\n').encode('utf-8')

                num = self.ser.write(input_s)
                self.data_num_sended += num
                self.lineEdit_2.setText(str(self.data_num_sended))
        else:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            print(data)
            num = len(data)
            # hex显示
            if self.hex_receive.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                self.s2__receive_text.insertPlainText(out_s)
                #print(out_s)
                decimal_number = int(out_s, 16)/255*5
                #print(decimal_number)

                data1[:-1] = data1[1:]  # shift data in the array one sample left

                data1[-1] = decimal_number

                data2.append(decimal_number)
                dataraw.append(int(out_s, 16))
                self.draw()


            else:
                # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
                self.s2__receive_text.insertPlainText(data.decode('iso-8859-1'))

            # 统计接收字符的数量
            self.data_num_received += num
            self.lineEdit.setText(str(self.data_num_received))

            # 获取到text光标
            textCursor = self.s2__receive_text.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去
            self.s2__receive_text.setTextCursor(textCursor)
        else:
            pass

    # 定时发送数据
    def data_send_timer(self):
        if self.timer_send_cb.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
        else:
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)

    # 清除显示
    def send_data_clear(self):
        self.s3__send_text.setText("")

    def receive_data_clear(self):
        self.s2__receive_text.setText("")

    def draw(self):
        self.watch.clear()
        plt1 = self.watch.addPlot(y=data2, title='电压')
        self.draw_change()
        self.watch_2.clear()
        pen = pg.mkPen({'color': '#ff0000', 'width': 0})

        self.watch_2.showGrid(x=True, y=True)
        # self.watch_2.setLogMode(x=False, y=False)

        plt1 = self.watch_2.plot(data1, pen=pen)
        print(1)
        # 创建一个Matplotlib Figure对象
        # print(1)
        # fig = plt.figure(figsize=(10, 6))
        # # 在Figure对象中创建一个Axes对象
        # ax = fig.add_subplot(111)
        # # 绘制一条曲线
        #
        # y = np.linspace(0, 5, 100)
        # ax.plot(data1, y)
        # # 创建一个FigureCanvas对象，并将Figure对象传入
        # canvas = FigureCanvas(fig)
        # # 获取pixmap对象
        # pixmap = QPixmap(canvas.grab().toImage())
        # # 设置背景

    def draw_change(self):
        print(1)

    def on_value_changed(self):
        if self.kedu.value() == 0:
            self.watch_2.setYRange(0, 0.1)
        if self.kedu.value() == 1:
            self.watch_2.setYRange(0, 0.5)
        if self.kedu.value() == 2:
            self.watch_2.setYRange(0, 1)
        if self.kedu.value() == 3:
            self.watch_2.setYRange(0, 2)
        if self.kedu.value() == 4:
            self.watch_2.setYRange(0, 5)
        if self.kedu.value() == 5:
            self.watch_2.setYRange(0, 10)

    def save(self):
        np.savetxt('dataraw.txt', dataraw,fmt='%d')

    def open_file_dialog(self):
        self.fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(
            self,
            '选择txt文件',
            '',
            'Text Files (*.txt)'
        )

        print(self.fileName)
    def showdata(self):
        txtdata = []

        with open(self.fileName, 'r') as file:
            for line in file:
                # 将每一行按照空格或制表符分割成单个字符串
                numbers = line.strip().split()
                print(numbers)
                # 将字符串转换为浮点数或整数，并添加到数组中
                for num in numbers:
                    try:
                        txtdata.append(float(num)/255*5)
                    except ValueError:
                        pass
        print(txtdata)
        self.watch.clear()
        plt1 = self.watch.addPlot(y=txtdata, title='电压')

        self.watch_2.clear()
        pen = pg.mkPen({'color': '#ff0000', 'width': 0})

        self.watch_2.showGrid(x=True, y=True)
        # self.watch_2.setLogMode(x=False, y=False)

        plt1 = self.watch_2.plot(txtdata, pen=pen)
        return txtdata

    def runpy(self):
        os.system("python onefile.py -c run")




if __name__ == '__main__':
    QGuiApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)



    myshow = Pyqt5_Serial()
    myshow.show()
    sys.exit(app.exec_())
