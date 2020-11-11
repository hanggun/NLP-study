### cross_entropy
在这里主要介绍一cross_entropy是怎么计算的

在NLP中，cross_entropy的计算如下：
$H(u,v)=E_u[-log(v(x))]=\sum_{x}u(x)log(v(x))$  
其中，u(x)是单词的真实分布，v(x)是模型的预测分布

在tensorflow中，计算cross_entropy可以使用ft.nn.sparse_softmax_cross_entropy_with_logits  
举一个例子：  
word_labels = tf.constant([2,0]) 假设词汇表大小为3  
predict_logits = tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])  
真实的单词分布假设为在词汇表中出现为1，不在词汇表中出现为0，因此，对于第一个单词2，只有在2的位置真实分布概率为1，其他位置概率分布为0，第二个单词与上述一致  
因此2个单词的loss计算为：
$\[1\timeslog(3/(2-1+3)),1\timeslog(1/(1+0-0.5))\]=\[0.32656264,0.46436879\]$ 
