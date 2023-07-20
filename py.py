

print('pyrun')

from PyQt5.QtWidgets import QApplication, QMessageBox

# 创建应用程序对象
app1 = QApplication([])

# 创建消息框
msg_box = QMessageBox()
msg_box.setWindowTitle("提示")
msg_box.setText("这是一个弹窗！")
msg_box.setIcon(QMessageBox.Information)

# 显示消息框
msg_box.exec_()

# 运行应用程序
# app1.exec_()