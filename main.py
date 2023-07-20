import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import tensorflow as tf
def load_data():
    # 读取数据
    x = np.load('train_data/train_data.npy')
    y = np.load('train_data/label.npy')
    num = len(Counter(y))
    print("类别数量为：", num)
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
    np.random.seed(99)
    np.random.shuffle(arr)
    # 按照打乱的顺序，重新排序
    arr_data = x[arr]
    arr_label = y[arr]
    # 将数据集分为训练集80%、测试集20%
    s = int(num_example * ratio)
    x_train = arr_data[:s]
    y_train = arr_label[:s]
    x_val = arr_data[s:]
    y_val = arr_label[s:]
    print("训练集shape", x_train.shape)
    print("训练集类别：", Counter(y_train))
    print("测试集shape", x_val.shape)
    print("测试集类别：", Counter(y_val))
    return x_train, y_train, x_val, y_val
# 读取数据
data, label, label_count = load_data()
# 生成训练集测试集,70%用作训练，30%用作测试
train_data, train_label, val_data, val_label = create_train_data(data, label, 0.7)
print("*"*10)
print("训练集数量：",len(train_label))
print("测试集数量：",len(val_label))
# 使用机器学习算法需要对多维数据进行降维
m_train = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
m_val = val_data.reshape(val_data.shape[0],val_data.shape[1]*val_data.shape[2])

# 设置训练迭代次数
epoch = 20
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score,classification_report,accuracy_score,log_loss
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn import model_selection
from sklearn.preprocessing import  OneHotEncoder

# 模型参数设置
rfc = RandomForestClassifier(n_estimators=50, min_samples_split=5, min_samples_leaf=4, max_depth=5)

# 模型准确率和损失值
acc_list = []
loss_list = []
train_acc_list = []
print("开始训练")
for i in range(1, epoch + 1):
    # 模型训练
    rfc.fit(m_train, train_label)
    # # 训练集
    # y_train = rfc.predict(m_train)
    # 测试集
    y_pred = np.asarray(rfc.predict(m_val), dtype='int64')
    # 计算准确率
    acc = round(accuracy_score(val_label, y_pred), 3)
    # 训练集
    y_pred = np.asarray(rfc.predict(m_train), dtype='int64')
    # 计算准确率
    train_acc = round(accuracy_score(train_label, y_pred), 3)
    # print('测试集准确率:', round(accuracy_score(val_label, y_pred),3))
    acc_list.append(acc)
    train_acc_list.append(train_acc)
    # 计算损失值
    # 使用one-hot编码计算损失值
    noe_hot = OneHotEncoder(sparse=False)
    y_pred_o = noe_hot.fit_transform(y_pred.reshape(1, -1))
    val_label_o = noe_hot.fit_transform(val_label.reshape(1, -1))
    #     loss = round(log_loss(val_label_o,y_pred_o),3)
    # print("loss：",round(log_loss(val_label,y_pred),3))
    #     loss_list.append(loss)
    print("完成第", i, "轮训练，测试集准确率：", acc)
    y_pred = np.asarray(rfc.predict(m_val), dtype='int64')
    print('------------------测试集上得分：------------------------')
    print('*' * 5)
    print('测试集准确率得分:', round(accuracy_score(val_label, y_pred), 3))
    print('*' * 5)
    print('准确率、召回率、f1-值测试报告如下:\n', classification_report(val_label, y_pred))
    # 设置Y轴范围
    plt.ylim(0, 1)
    # 训练准确率曲线
    show_data1 = train_acc_list
    # 测试准确率曲线
    show_data2 = acc_list
    x_data = list(range(1, len(show_data1) + 1))
    ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
    ln2, = plt.plot(x_data, show_data2, color='red', linewidth=3.0, linestyle='-.')
    plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
    plt.show()
    from joblib import dump, load
    # 保存模型
    dump(rfc, 'models_save/rfc.joblib')
    # # 加载
    # rfc = load('models_save/rfc.joblib')
    # print(rfc)
    from tensorflow import keras
    from tensorflow.keras import layers, models
    # 查看GPU是否可用
    print(tf.test.is_gpu_available())
    def cnn_create():
        loss = 'sparse_categorical_crossentropy'
        metrics = ['acc']
        inputs = keras.Input(shape=(1000, 2))
        x = layers.Conv1D(64, 3, activation='relu')(inputs)
        x = layers.MaxPooling1D(16)(x)
        # 全局平均池化GAP层
        x = layers.GlobalAveragePooling1D()(x)
        # 几个密集分类层
        x = layers.Dense(32, activation='relu')(x)
        # 退出层
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(4, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss=loss,
                      metrics=metrics)
        print("实例化模型成功，参数如下：")
        print(model.summary())
        return model
        # 实例化模型
    cnn_model = cnn_create()
    # 设置模型log输出地址
    log_dir = os.path.join("logs/CNN")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        # 模型训练
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = cnn_model.fit(train_data, train_label, epochs=20, batch_size=16, validation_split=0.2,
                                callbacks=[tensorboard_callback])
        from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, log_loss
        print("*****完成预处理，进行模型评估*****")
        y_pred = cnn_model.predict(val_data)
        y_pred = [np.argmax(x) for x in y_pred]
        print('------------------测试集上得分：------------------------')
        print('*' * 5)
        print('测试集准确率得分:', round(accuracy_score(val_label, y_pred), 3))
        print('*' * 5)
        print('准确率、召回率、f1-值测试报告如下:\n', classification_report(val_label, y_pred))
        # 设置Y轴范围
        plt.ylim(0, 1)
        # 训练准确率曲线
        show_data1 = history.history['acc']
        # 测试准确率曲线
        show_data2 = history.history['val_acc']
        x_data = list(range(1, len(show_data1) + 1))
        ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
        ln2, = plt.plot(x_data, show_data2, color='red', linewidth=3.0, linestyle='-.')
        plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
        plt.show()
        # 保存模型
        model_path = 'models_save/cnn_model.h5'
        cnn_model.save(model_path)
        print("完成模型训练，保存地址：", model_path)
        import n_model as md
        import tensorflow as tf

        # 模型参数
        model_param = {
            "a_shape": 1000,
            "b_shape": 2,
            "label_count": 4,
            "num_b": 5
        }

        data_shape = (model_param['a_shape'], model_param['b_shape'])
        # 模型实例化
        model = md.CNN_ResNet_model(model_param['label_count'], model_param['num_b'], data_shape=data_shape)
        # 使用学习率进行训练
        res_model = model.model_create(learning_rate=1e-4)
        # 模型网络结构
        print("实例化模型成功，网络结构如下：")
        print(res_model.summary())
        # 设置模型log输出地址
        log_dir = os.path.join("logs/ResNet")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            # 模型训练
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            history = res_model.fit(train_data, train_label, epochs=20, batch_size=32, validation_split=0.2,
                                    callbacks=[tensorboard_callback])
            from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, log_loss

            print("*****完成预处理，进行模型评估*****")
            y_pred = res_model.predict(val_data)
            y_pred = [np.argmax(x) for x in y_pred]
            print('------------------测试集上得分：------------------------')
            print('*' * 5)
            print('测试集准确率得分:', round(accuracy_score(val_label, y_pred), 3))
            print('*' * 5)
            print('准确率、召回率、f1-值测试报告如下:\n', classification_report(val_label, y_pred))
            # 设置Y轴范围
            plt.ylim(0, 1)
            # 训练准确率曲线
            show_data1 = history.history['acc']
            # 测试准确率曲线
            show_data2 = history.history['val_acc']
            x_data = list(range(1, len(show_data1) + 1))
            ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
            ln2, = plt.plot(x_data, show_data2, color='red', linewidth=3.0, linestyle='-.')
            plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
            plt.show()
            # 保存模型
            model_path = 'models_save/cnn_resnet_model.h5'
            res_model.save(model_path)
            print("完成模型训练，保存地址：", model_path)