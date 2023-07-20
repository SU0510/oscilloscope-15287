from collections import Counter
from tensorflow import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
# 加载模型
model = keras.models.load_model('D:\pycharm\pythonProject\models_save\cnn_resnet_model.h5')
#print(model.summary())
# 使用模型进行预测
def load_data():
    # 读取数据
    x = np.load('train_data/train_data.npy')
    y = np.load('train_data/label.npy')
    num = len(Counter(y))
    return x, y, num
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
# 读取数据
data, label, label_count = load_data()
data,label=create_train_data(data,label,1.0)
# 生成训练集测试集,70%用作训练，30%用作测试
#这段代码应该可以用来直接读取原始数据
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
'''''''''
data_97,label_97 = data_load('data/Normal Baseline Data/',97,1000,0)
data_98,label_98 = data_load('data/Normal Baseline Data/',98,1000,0)
data_99,label_99 = data_load('data/Normal Baseline Data/',99,1000,0)
data_100,label_100 = data_load('data/Normal Baseline Data/',100,1000,0)
data_normal = data_97 + data_98 + data_99 + data_100
label_normal = label_97 + label_98 + label_99 +label_100
data_105,label_105 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',105,1000,1)
data_106,label_106 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',106,1000,1)
data_107,label_107 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',107,1000,1)
data_108,label_108 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',108,1000,1)
data_169,label_169 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',169,1000,1)
data_170,label_170 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',170,1000,1)
data_171,label_171 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',171,1000,1)
data_172,label_172 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',172,1000,1)
data_209,label_209 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',209,1000,1)
data_210,label_210 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',210,1000,1)
data_211,label_211 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',211,1000,1)
data_212,label_212 = data_load('data/12k Drive End Bearing Fault Data/内圈故障/',212,1000,1)
data_inner = data_105 + data_106 + data_107 + data_108 + data_169 + data_170 + data_171 + data_172 + data_209 + data_210 + data_211 + data_212
label_inner = label_105 + label_106 + label_107 + label_108 + label_169 + label_170 + label_171 + label_172 + label_209 + label_210 + label_211 + label_212
print("处理后内圈样本shape：",np.asarray(data_inner,dtype = 'float').shape)
print("label数：",len(label_inner))
data_118,label_118 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',118,1000,2)
data_119,label_119 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',119,1000,2)
data_120,label_120 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',120,1000,2)
data_121,label_121 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',121,1000,2)
data_185,label_185 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',185,1000,2)
data_186,label_186 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',186,1000,2)
data_187,label_187 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',187,1000,2)
data_188,label_188 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',188,1000,2)
data_222,label_222 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',222,1000,2)
data_223,label_223 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',223,1000,2)
data_224,label_224 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',224,1000,2)
data_225,label_225 = data_load('data/12k Drive End Bearing Fault Data/滚动体故障/',225,1000,2)
data_ball = data_118 + data_119 + data_120 + data_121 + data_185 + data_186 + data_187 + data_188 + data_222 + data_223 + data_224 + data_225
label_ball = label_118 + label_119 + label_120 + label_121 + label_185 + label_186 + label_187 + label_188 + label_222 + label_223 + label_224 + label_225
data_130,label_130 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',130,1000,3)
data_131,label_131 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',131,1000,3)
data_132,label_132 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',132,1000,3)
data_133,label_133 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',133,1000,3)
data_197,label_197 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',197,1000,3)
data_198,label_198 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',198,1000,3)
data_199,label_199 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',199,1000,3)
data_200,label_200 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',200,1000,3)
data_234,label_234 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',234,1000,3)
data_235,label_235 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',235,1000,3)
data_236,label_236 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',236,1000,3)
data_237,label_237 = data_load('data/12k Drive End Bearing Fault Data/外圈故障/',237,1000,3)
data_outer = data_130 + data_131 + data_132 + data_133 + data_197 + data_198 + data_199 + data_200 + data_234 + data_235 + data_236 + data_237
label_outer = label_130 + label_131 + label_132 + label_133 + label_197 + label_198 + label_199 + label_200 + label_234 + label_235 + label_236 + label_237
data_train = np.asarray(data_normal[:1400] + data_inner[:1400] + data_ball[:1400] + data_outer[:1400],dtype = 'float64')
label = np.asarray(label_normal[:1400] + label_inner[:1400] + label_ball[:1400] + label_outer[:1400],dtype = 'int64')
print("处理后样本shape：",data_train.shape)
print("处理后数据类别分布：",Counter(label))
# 保存数据
np.save("train_data/train_data.npy",data_train)
np.save("train_data/label.npy",label)
print("数据保存成功，位置：/train_data/")
'''''''''
input_data = [data]  # 输入你的数据
predictions = model.predict(input_data)
print("预测结果:", predictions)
# 假设模型的预测输出为 pred，是一个形状为 (batch_size, num_classes) 的 numpy 数组
pred = np.array(predictions)  # 示例数据，可以根据实际情况进行修改
# 获取概率最高的类别索引
predicted_index = np.argmax(pred, axis=1)
# 输出最终判断出的类别和对应的索引值
print("预测类别：", predicted_index)
import tkinter as tk
from tkinter import ttk
def display_results(data):
    root = tk.Tk()
    # 创建表格控件
    table = ttk.Treeview(root, columns=('Number', 'Status'))
    table.heading('#0', text='第几个')
    table.heading('Number', text='故障代号')
    table.heading('Status', text='Status')
    for i, number in enumerate(data, start=1):
        status = get_status(number)
        table.insert('', 'end', text=str(i), values=(number, status))
    table.pack()
    root.mainloop()

def get_status(number):
    status_dict = {0: '正常', 1: '内圈故障', 2: '外圈故障', 3: '滚动体故障'}
    return status_dict.get(number, '未知状态')

display_results(predicted_index)