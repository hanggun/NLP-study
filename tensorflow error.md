### ValueError: No gradients provided for any variable  
问题原因1：计算loss的变量未参与反向传播
###  Could not interpret initializer identifier
```python
store = layers.Embedding(2,4, embeddings_initializer=np.ones([2,4]))
```
此问题是因为没有使用初始器，添加初始器就可以了
```pyhton
store = layers.Embedding(2,4, embeddings_initializer=keras.initializers.Constant(np.ones([2,4])))
```
