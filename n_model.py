import tensorflow
from tensorflow import  keras
from  tensorflow.keras import  layers,models

class CNN_ResNet_model:
    def __init__(self, label_num , num_b , data_shape=(1000, 2)):  # 默认输入张量为(1000,2)
        # res块数量
        self.num_blocks = num_b
        # 过滤器数量
        self.filters = 64
        # 步长
        self.conv_size = 3
        # 分类类别数
        self.label_num = label_num
        # 数据输入的shape
        self.data_shape = data_shape
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['acc']

    def res_net_block(self, input_data):
        # CNN层
        x = layers.Conv1D(self.filters, self.conv_size, activation='relu', padding='same')(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(self.filters, self.conv_size, activation=None, padding='same')(x)
        # 第二层没有激活函数
        x = layers.BatchNormalization()(x)
        # 两个张量相加
        x = layers.Add()([x, input_data])
        # 对相加的结果使用ReLU激活
        x = layers.Activation('relu')(x)
        # 返回结果
        return x

    def model_create(self, learning_rate):
        inputs = keras.Input(shape=(self.data_shape[0], self.data_shape[1]))
        x = layers.Conv1D(32, 3, activation='relu')(inputs)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPooling1D(16)(x)
        # 定义res层
        for i in range(self.num_blocks):
            x = self.res_net_block(x)
        # 添加一个CNN层
        x = layers.Conv1D(64, 3, activation='relu')(x)
        # 全局平均池化GAP层
        x = layers.GlobalAveragePooling1D()(x)
        # 几个密集分类层
        x = layers.Dense(256, activation='relu')(x)
        # 退出层
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.label_num, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss=self.loss,
                      metrics=self.metrics)
        # print(model.summary())
        return model
