## Target
此任务的目的是对多标签任务进行分类，每一个样本有固定的两类标签，且两类标签存在关联，第一类标签有4种，第二类标签有27种。

### 学习心得
在这个任务中，学习了如何灵活使用keras进行构建网络

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
