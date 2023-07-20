from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import Counter
# 定义数据分布展示

def show_plot(data,show_num):
    data_list = []
    for a in data[:show_num]:
        data_list.append(a[0])
    return data_list
m_97 = loadmat('data/Normal Baseline Data/97.mat')
print(m_97)
print('\n')
print(m_97['X097_DE_time'])
print('\n')
print(m_97['X097_FE_time'])
print('\n')
print(m_97['X097_DE_time'][0])
print('\n')
print('标准数据长度：',len(m_97['X097_DE_time']))
print('电机转速：',m_97['X097RPM'][0][0])
# 设置Y轴范围
plt.ylim(-0.5,0.5)
# 抽样1000个进行展示
show_x = show_plot(m_97['X097_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_98 = loadmat('data/Normal Baseline Data/98.mat')
m_98
print('标准数据长度：',len(m_98['X098_DE_time']))
# 设置Y轴范围
plt.ylim(-0.5,0.5)
# 抽样1000个进行展示
show_x = show_plot(m_98['X098_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_99 = loadmat('data/Normal Baseline Data/99.mat')
m_99
print('标准数据长度：',len(m_99['X099_DE_time']))
# 设置Y轴范围
plt.ylim(-0.5,0.5)
# 抽样1000个进行展示
show_x = show_plot(m_99['X099_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_100 = loadmat('data/Normal Baseline Data/100.mat')
m_100
print('标准数据长度：',len(m_100['X100_DE_time']))
print('电机转速：',m_100['X100RPM'][0][0])
# 设置Y轴范围
plt.ylim(-0.5,0.5)
# 抽样1000个进行展示
show_x = show_plot(m_100['X100_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_105 = loadmat('data/12k Drive End Bearing Fault Data/内圈故障/105.mat')
print('标准数据长度：',len(m_105['X105_DE_time']))
# 设置Y轴范围
plt.ylim(-0.8,0.8)
# 抽样1000个进行展示
show_x = show_plot(m_105['X105_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_118 = loadmat('data/12k Drive End Bearing Fault Data/滚动体故障/118.mat')
print(m_118)
print('标准数据长度：',len(m_118['X118_DE_time']))
# 设置Y轴范围
plt.ylim(-0.5,0.5)
# 抽样1000个进行展示
show_x = show_plot(m_118['X118_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_119 = loadmat('data/12k Drive End Bearing Fault Data/滚动体故障/119.mat')
print('标准数据长度：',len(m_119['X119_DE_time']))
# 设置Y轴范围
plt.ylim(-0.5,0.5)
# 抽样1000个进行展示
show_x = show_plot(m_119['X119_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
m_130 = loadmat('data/12k Drive End Bearing Fault Data/外圈故障/130.mat')
print('标准数据长度：',len(m_130['X130_DE_time']))
# 设置Y轴范围
plt.ylim(-0.8,0.8)
# 抽样1000个进行展示
show_x = show_plot(m_130['X130_DE_time'],1000)
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()

# 定义样本分割函数
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
data_99,label_99 = data_load('data/Normal Baseline Data/',99,1000,0)
data_100,label_100 = data_load('data/Normal Baseline Data/',100,1000,0)
# 设置Y轴范围
plt.ylim(0,1)
show_x = data_97[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
# 设置Y轴范围
plt.ylim(0,1)
show_x = data_98[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_99[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_100[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
print(np.asarray(data_97,dtype = 'float').shape)
print(np.asarray(data_98,dtype = 'float').shape)
print(np.asarray(data_99,dtype = 'float').shape)
print(np.asarray(data_100,dtype = 'float').shape)
data_normal = data_97 + data_98 + data_99 + data_100
label_normal = label_97 + label_98 + label_99 +label_100
print("处理后正常样本shape：",np.asarray(data_normal,dtype = 'float').shape)
print("label数：",len(label_normal))
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
# 设置Y轴范围
plt.ylim(0,1)
show_x = data_209[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
# 设置Y轴范围
plt.ylim(0,1)
show_x = data_210[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_211[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_212[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
print(np.asarray(data_105,dtype = 'float').shape)
print(np.asarray(data_106,dtype = 'float').shape)
print(np.asarray(data_107,dtype = 'float').shape)
print(np.asarray(data_108,dtype = 'float').shape)
print(np.asarray(data_169,dtype = 'float').shape)
print(np.asarray(data_170,dtype = 'float').shape)
print(np.asarray(data_171,dtype = 'float').shape)
print(np.asarray(data_172,dtype = 'float').shape)
print(np.asarray(data_209,dtype = 'float').shape)
print(np.asarray(data_210,dtype = 'float').shape)
print(np.asarray(data_211,dtype = 'float').shape)
print(np.asarray(data_212,dtype = 'float').shape)
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
# 设置Y轴范围
plt.ylim(0,1)
show_x = data_118[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_119[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_120[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_121[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
data_ball = data_118 + data_119 + data_120 + data_121 + data_185 + data_186 + data_187 + data_188 + data_222 + data_223 + data_224 + data_225
label_ball = label_118 + label_119 + label_120 + label_121 + label_185 + label_186 + label_187 + label_188 + label_222 + label_223 + label_224 + label_225
print("处理后滚动体样本shape：",np.asarray(data_ball,dtype = 'float').shape)
print("label数：",len(label_ball))
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
# 设置Y轴范围
plt.ylim(0,1)
show_x = data_130[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_131[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_132[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()# 设置Y轴范围
plt.ylim(0,1)
show_x = data_133[0]
x = range(len(show_x))
plt.plot(x,show_x)
plt.show()
data_outer = data_130 + data_131 + data_132 + data_133 + data_197 + data_198 + data_199 + data_200 + data_234 + data_235 + data_236 + data_237
label_outer = label_130 + label_131 + label_132 + label_133 + label_197 + label_198 + label_199 + label_200 + label_234 + label_235 + label_236 + label_237
print("处理后外圈样本shape：",np.asarray(data_outer,dtype = 'float').shape)
print("label数：",len(label_outer))
data_train = np.asarray(data_normal[:1400] + data_inner[:1400] + data_ball[:1400] + data_outer[:1400],dtype = 'float64')
label = np.asarray(label_normal[:1400] + label_inner[:1400] + label_ball[:1400] + label_outer[:1400],dtype = 'int64')
print("处理后样本shape：",data_train.shape)
print("处理后数据类别分布：",Counter(label))
# 保存数据
np.save("train_data/train_data.npy",data_train)
np.save("train_data/label.npy",label)
print("数据保存成功，位置：/train_data/")