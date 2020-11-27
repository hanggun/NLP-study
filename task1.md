## Target
此任务的目的是对多标签任务进行分类，每一个样本有固定的两类标签，且两类标签存在关联，第一类标签有4种，第二类标签有27种。

### 学习心得
在这个任务中，学习了机器学习基本知识的应用以及学习了如何灵活的使用keras进行构建网络
#### batch归一化
在机器学习中，一个很重要的点是对数据进行归一化，当数据差异很大时，使用梯度下降法需要经过更多轮的迭代，并且走的路线可能是曲折的，而归一化之后，数据之间的差异变小，走的路线也可能会变成笔直的，从而加速了梯度下降法。[参考](https://www.cnblogs.com/bonelee/p/7124695.html)
#### 自定义loss
在这里展示了3种自定义loss的使用方法，在使用函数式API添加loss的时候，我一度觉得Model.add_loss()方法无法添加损失函数，事实是完全可以！添加的loss必须为值而不是列表或其他形式。
```python
'''简单的使用函数
'''
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
          1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.math.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.math.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss
```
```python
'''继承keras.losses.Loss类
'''
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor
```
```python
'''继承keras.layers.Layer类，在添加loss的同时添加指标跟踪
'''
class myLoss(keras.layers.Layer):
    def __init__(self, name=None):
        super(myLoss, self).__init__(name=name)
        self.loss_fn = keras.losses.CategoricalCrossentropy()
        self.accuracy_first = keras.metrics.CategoricalAccuracy(name='acc_first_label')
        self.accuracy_second = keras.metrics.CategoricalAccuracy(name='acc_second_label')

    def call(self, input_first_label,pred_first,input_second_label,pred_second, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(input_first_label, pred_first)
        loss += self.loss_fn(input_second_label, pred_second)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc_first_label = self.accuracy_first(input_first_label, pred_first)
        acc_second_label = self.accuracy_second(input_second_label, pred_second)
        self.add_metric(acc_first_label, name="acc_first_label")
        self.add_metric(acc_second_label, name="acc_second_label")

        # Return the inference-time prediction tensor (for `.predict()`).
        return pred_second
        
 #使用这种方法返回的是输出，因此可以放入Model类中使用
 loss = myLoss(name='myLoss')(input_first_label,pred_first,input_second_label,pred_second)
 model = keras.Model(inputs=[input_ids, input_first_label, input_second_label], outputs=loss)
```
focal loss, 可以处理类别不平衡分类问题的损失，其主要思想是让模型着重训练hard example，忽略easy example，不过在我的项目中未能提升精度，这边记录一下在keras中的实现方式
```python
loss = -(1 - y_pred) ** 2 * y_true * tf.math.log(y_pred)
```
互信息损失，其主要思想是将每个类别出现的频率作为先验分布，添加到交叉熵中，解决类别不平衡的重点，在多标签模型中，效果仍然不理想，[参考](https://kexue.fm/archives/7615)
```python
def categorical_crossentropy_with_prior(y_true, y_pred, tau=1.0):
    """带先验分布的交叉熵
    注：y_pred不用加softmax
    """
    prior = xxxxxx  # 自己定义好prior，shape为[num_classes]
    log_prior = K.constant(np.log(prior + 1e-8))
    for _ in range(K.ndim(y_pred) - 1):
        log_prior = K.expand_dims(log_prior, 0)
    y_pred = y_pred + tau * log_prior
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
```
#### 自定义指标
```python
'''继承keras.metrics.Metric类
这里计算的是真实标签与预测标签中概率最大的前两位的准确率
'''
class myAccuracy(keras.metrics.Metric):
    def __init__(self, name="myAccuracy", **kwargs):
        super(myAccuracy, self).__init__(name=name, **kwargs)
        self.true_rate = self.add_weight(name="ma", dtype=tf.float32, initializer="zeros")
        self.total = self.add_weight(name='total', dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argsort(y_pred, axis=1, direction='DESCENDING')
        y_true = tf.argsort(y_true, axis=1, direction='DESCENDING')
        values = tf.logical_and(tf.logical_or(y_pred[:,0]==y_true[:,0],y_pred[:,0]==y_true[:,1]),
                                tf.logical_or(y_pred[:,1]==y_true[:,0],y_pred[:,1]==y_true[:,1]))
        values = tf.reduce_sum(tf.cast(values, "float32"))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], 'float32'))
        self.true_rate.assign_add(values)

    def result(self):
        return self.true_rate / self.total

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_rate.assign(0.0)
        self.total.assign(0.0)
```

#### 使用Lambda快速添加简单层
```python
features=layers.Lambda(lambda x: K.concatenate([x[0], x[1]], 1))([feature,input_first_embedding])
```

#### Embedding层的妙用
1.作为一个矩阵查找操作，输入一个整数，输出对应下标的向量，并且这个矩阵是可以训练的
```python
store = layers.Embedding(2,4, embeddings_initializer=keras.initializers.Constant(np.array([[0,0,0,0],[1,1,1,1]])), trainable=False)
store(1)
#输出<tf.Tensor: id=10065, shape=(4,), dtype=float32, numpy=array([1., 1., 1., 1.], dtype=float32)>
```
2.将输入的最后一个维度的每一个数转换成output_dim维向量，并使用embedding_initializer进行初始化，default为'uniform'
```python
store = layers.Embedding(input_dim=2,output_dim=4)#input_dim为输入的整数的range为[0,input_dim),output_dim为输出维度
store(tf.ones(shape=[2,4]))
'''输出<tf.Tensor: id=10116, shape=(2, 4, 4), dtype=float32, numpy=
array([[[-0.01051073,  0.01206005,  0.00975666, -0.03649573],
        [-0.01051073,  0.01206005,  0.00975666, -0.03649573],
        [-0.01051073,  0.01206005,  0.00975666, -0.03649573],
        [-0.01051073,  0.01206005,  0.00975666, -0.03649573]],

       [[-0.01051073,  0.01206005,  0.00975666, -0.03649573],
        [-0.01051073,  0.01206005,  0.00975666, -0.03649573],
        [-0.01051073,  0.01206005,  0.00975666, -0.03649573],
        [-0.01051073,  0.01206005,  0.00975666, -0.03649573]]],
      dtype=float32)>
 '''
```

#### 多输入多输出模型构建
```python
'''不自定义loss和metrics的基础模型
'''
inputs = layers.Input(shape=(4))
inputs1 = layers.Input(shape=(4))
x = layers.Dense(4, activation='softmax',name='x')(inputs)
y = layers.Dense(2, activation='softmax',name='y')(inputs1)
model = keras.Model([inputs,inputs1], [x,y])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
             loss=[
                 keras.losses.CategoricalCrossentropy(),
                 lambda y_true, y_pred: y_pred
             ],
             metrics=['acc', 'acc']
                )
x_train = np.array([[1,0,1,0], [0,1,0,1]])
x_valid = np.array([[0,0,0,1],[0,0,1,0]])
y1 = np.array([[0,0,0,1],[0,0,1,0]])
y2 = np.array([[0,1],[1,0]])
model.fit([x_train, x_valid], [y1,y2], epochs=4)
#输出Epoch 1/4
#2/2 [==============================] - 0s 160ms/sample - loss: 1.3203 - x_loss: 0.8203 - y_loss: 0.5000 - x_acc: 1.0000 - y_acc: 0.5000
#既可以看到各自的loss和各自的准确率，还能看到总的loss
```
```python
'''自定义loss和metrics
'''
def loss1(y_true, y_pred):
    return K.mean(y_true) - K.mean(y_pred)

def loss2(y_true, y_pred):
    return K.mean(y_true) + K.mean(y_pred)

def metric1(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)

def metric2(y_true, y_pred):
    return tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
    
inputs = layers.Input(shape=(4))
inputs1 = layers.Input(shape=(4))
x = layers.Dense(4, activation='sigmoid',name='x')(inputs)
y = layers.Dense(2, activation='softmax',name='y')(inputs1)
model = keras.Model([inputs,inputs1], [x,y])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
             loss=[
                 loss1,
                 loss2
             ],
              weight=[1,1],
             metrics={'x':metric1,'y':metric2}
                )
x_train = np.array([[1,0,1,0], [0,1,0,1]], dtype='float')
x_valid = np.array([[0,0,0,1],[0,0,1,0]], dtype='float')
y1 = np.array([[0,0,0,1],[0,0,1,0]], dtype='float')
y2 = np.array([[0,1],[1,0]], dtype='float')
model.fit([x_train, x_valid], [y1,y2], epochs=4)
#输出Epoch 1/4
#2/2 [==============================] - 0s 181ms/sample - loss: -0.1273 - x_loss: -0.2057 - y_loss: 0.0783 - x_metric1: 0.6250 - y_metric2: 0.3102
#loss和之前一样，对loss进行了加权，loss分别作用于x层和y层
```
上述两种多输入多输出的模型都是在匹配了预测和真实标签的情况下进行构建的，但是在某些时候，我们希望将真实标签作为输入进行模型构建，这时候就无法再compile中使用loss和metrics，需要在模型外自定义loss和metrics，以下为基础模型例子：
```python
inputs = layers.Input(shape=(4))
inputs1 = layers.Input(shape=(4))
true_x = layers.Input(shape=(None,))
true_y = layers.Input(shape=(None,))
x = layers.Dense(4, activation='softmax',name='x')(inputs)
y = layers.Dense(2, activation='softmax',name='y')(inputs1)
model = keras.Model([inputs,inputs1, true_x, true_y], [x,y])

loss1 = K.categorical_crossentropy(true_x, x)
loss2 = K.binary_crossentropy(true_y, y)
loss = K.mean(loss1) + K.mean(loss2)
model.add_loss(loss)

acc1 = tf.keras.metrics.categorical_accuracy(true_x, x)
acc2 = tf.keras.metrics.binary_accuracy(true_y, y)
model.add_metric(acc1, name='x_acc', aggregation='mean')
model.add_metric(acc2, name='y_acc', aggregation='mean')

model.compile(optimizer=keras.optimizers.Adam(1e-3))
x_train = np.array([[1,0,1,0], [0,1,0,1]], dtype='float')
x_valid = np.array([[0,0,0,1],[0,0,1,0]], dtype='float')
y1 = np.array([[0,0,0,1],[0,0,1,0]], dtype='float')
y2 = np.array([[0,1],[1,0]], dtype='float')
model.fit([x_train, x_valid, y1, y2], epochs=4)
#输出Epoch 1/4
#2/2 [==============================] - 0s 185ms/sample - loss: 1.8142 - x_acc: 1.0000 - y_acc: 0.5000
#可以追踪总体损失但是不能追踪各个损失
```

#### 模型的断点设置
使用keras.callbacks.ModelCheckpoint放入model.fit的callback参数中即可，需要注意的是，如果模型中存在自定义的损失和指标，则在加载模型的时候，需要在custom_objects参数中输入自定义的损失和指标，并且目前测试下来，无法使用savedmodel格式加载自定义损失，最好使用'.h5'形式的格式。加入了对样本的各个类别进行了加权，权重直接计算在损失当中，无法对多输入进行加权，因此在这里只使用了单输入的情况
```python
'''加入了断点设置，以及class_weight的使用方法
'''
tf.random.set_seed(1)
inputs = layers.Input(shape=(4))
x = layers.Dense(2, activation='sigmoid',name='x')(inputs)
model = keras.Model(inputs, x)

x_train = np.random.randint(4, size = [10,1])
enc.fit(np.arange(4).reshape(-1,1))
x_train = enc.transform(x_train).toarray()

y1 = np.random.randint(2, size = [10,1])
enc.fit(np.arange(2).reshape(-1,1))
y1 = enc.transform(y1).toarray()

model.compile(optimizer=keras.optimizers.Adam(1e-3),
             loss=[
                 myloss
             ],
             metrics=['acc']
                )

class_weight = {
    0: 1,
    1: 4,
}
callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel.h5",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
]
model.fit(x_train, y1, class_weight = class_weight, validation_split = 0.1,epochs=10, callbacks=callbacks)
model = keras.models.load_model('mymodel.h5', custom_objects={'myloss': myloss})
```
#### 使用tensorboard观察
tensorboard首先需要添加keras.callbacks.Tensorboard回调函数，在添加路径的时候，需要将回调函数文件路径加入到系统路径，具体可以在代码中发现，在写完了tensorboard日志之后，我们可以在命令行输入`tensorboard --logdir log_path`运行，最后在http://localhost:6006 中进行查看
```python
logdir = os.path.join("tensorboard_log")
if not os.path.exists(logdir):
    os.mkdir(logdir)

tensorboard = keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",
)
history=model.fit(train_x,train_y,epochs=cfg.num_epochs,batch_size=cfg.batch_size,
                  verbose=1, validation_split=0.2, callbacks=[tensorboard])
```
