import time
from collections import Counter
from tensorflow import keras
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
model = keras.models.load_model('D:\pycharm\pythonProject\models_save\cnn_resnet_model.h5')
'''''''''
def data_load(path, data_name, cut_num, label):
    """
    path：数据地址
    data_name：数据名称
    cut_num：每份样本数量
    label:数据标签
    """
    name_str = str(data_name)
    data = loadmat(path + name_str + '.mat')
    # 返回list
    list_r = []
    if data_name < 100:
        data_name = '0' + str(data_name)
    else:
        data_name = str(data_name)
    # 原始数据
    org_DE = data['X' + data_name + '_DE_time']
    org_FE = data['X' + data_name + '_FE_time']
    # 数据归一化
    # 归一化DE
    scaler = MinMaxScaler()
    #     scaler.fit(list_DE)
    list_DE_n = scaler.fit_transform(org_DE)
    # 归一化FE
    scaler = MinMaxScaler()
    #     scaler.fit(list_FE)
    list_FE_n = scaler.fit_transform(org_FE)
    # 构建一维数组
    list_DE = []
    for de in list_DE_n:
        list_DE.append(de[0])
    list_FE = []
    for fe in list_FE_n:
        list_FE.append(fe[0])
    # 将de，fe数据放入list
    for de, fe in zip(list_DE, list_FE):
        list_r.append([de, fe])
    data_cut = []
    label_cut = []
    # 分割数据
    for i in range(0, int(len(list_DE_n) / cut_num)):
        data_cut.append(list_r[i * cut_num: (i + 1) * cut_num])
        label_cut.append(label)
    return data_cut, label_cut
data_97,label_97 = data_load('data/Normal Baseline Data/',97,1000,0)
data_98,label_98 = data_load('data/Normal Baseline Data/',98,1000,0)
data_normal = data_97
data_train = np.asarray(data_normal[:1400],dtype='float64')
label_normal = label_97
label = np.asarray(label_normal[:1400],dtype='int64')
np.save("one file/train_data.npy",data_train)
np.save("one file/label.npy",label)
print("数据保存成功，位置：/one file/")
print("处理后样本shape：",data_train.shape)
print("处理后数据类别分布：",Counter(label))
'''
def create_train_data(x, y, ratio=0.8):
    """
    x:数据
    y:类别
    ratio:生成训练集比率
    """
    # 打乱顺序
    # 读取data矩阵的第一维数（图片的个数）
    num_example = x.shape[0]
    # 产生一个num_example范围，步长为1的序列
    arr = np.arange(num_example)
    # 调用函数，打乱顺序
    np.random.seed(int(time.time()))
    np.random.shuffle(arr)
    # 按照打乱的顺序，重新排序
    arr_data = x[arr]
    arr_label = y[arr]
    # 将数据集分为训练集80%、测试集20%
    s = int(num_example * ratio)
    x_train = arr_data[:s]
    y_train = arr_label[:s]
    return x_train, y_train
def load_data():
    # 读取数据
    x = np.load('one file/train_data.npy')
    y = np.load('one file/label.npy')
    num = len(Counter(y))
    return x, y, num
# 读取数据
data, label, label_count = load_data()
data,label=create_train_data(data,label,1.0)
input_data = [data]  # 输入你的数据
predictions = model.predict(input_data)
print("预测结果:", predictions)
# 假设模型的预测输出为 pred，是一个形状为 (batch_size, num_classes) 的 numpy 数组
pred = np.array(predictions)  # 示例数据，可以根据实际情况进行修改
# 获取概率最高的类别索引
predicted_index = np.argmax(pred, axis=1)
import tkinter as tk
from tkinter import ttk
def display_result(result):
    status_dict = {0: '正常', 1: '内圈故障', 2: '外圈故障', 3: '滚动体故障'}
    result_status = status_dict.get(result, '未知状态')
    root = tk.Tk()
    label = ttk.Label(root, text="预测结果：{}".format(result_status))
    label.pack()
    root.mainloop()

# 获取概率最高的类别索引
predicted_index = np.argmax(pred, axis=1)[0]
# 输出预测类别和对应的状态
display_result(predicted_index)