### cross_entropy
在这里主要介绍一cross_entropy是怎么计算的

在NLP中，cross_entropy的计算如下：
$H(u,v)=E_u[-log(v(x))]=-\sum_{x}u(x)log(v(x))$  
其中，u(x)是单词的真实分布，v(x)是模型的预测分布

在tensorflow中，计算cross_entropy可以使用ft.nn.sparse_softmax_cross_entropy_with_logits  
举一个例子：  
word_labels = tf.constant([2,0]) 假设词汇表大小为3  
predict_logits = tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])  
softmax转换logits到概率的方程如下：  
$$p_t=\frac{e^{z_t}}{\sum_{k}e^{z_k}}$$
第一个单词label为2的概率可以计算为:  
$$\frac{e^3}{e^3+e^{-1}+e^2}=0.7213$$
真实的单词分布假设为在词汇表中出现为1，不在词汇表中出现为0，因此，对于第一个单词2，只有在2的位置真实分布概率为1，其他位置概率分布为0，第二个单词与上述一致  
因此2个单词的loss计算为：
$\[-1\times log(p_{02}),-1\times log(p_{10})\]=\[0.32656264,0.46436879\]$  
其中$p_{02}$代表第一个单词标签为2的概率，$p_{10}$代表第一个单词标签为0的概率

### softmax求导
$$(\frac{e^{z_t}}{\sum_{k}e^{z_k}})'=\frac{\sum_{k}e^{z_k}e^{z_t}-(e^{z_t})^2}{(\sum_{k}e^{z_k})^2}=\frac{e^{z_t}}{\sum_{k}e^{z_k}}(1-\frac{e^{z_t}}{\sum_{k}e^{z_k}})=p_t(1-p_t)$$

### sigmoid求导
softmax和sigmoid有着相同的求导公式  
$$(\frac{1}{1+e^{-x}})'=\frac{e^{-x}}{(1+e^{-x})^2}=f(x)(1-f(x))$$
### backpropagation(误差逆传播算法)
误差逆传播的思想是通过将预测结果与真实结果之间的误差与每一个神经元拥有的权重联系起来，通过梯度下降法等算法对权重进行调整，使最终的预测结果与真实结果之间的误差减小，权重的迭代公式如下所示
$$\omega = \omega-\alpha\frac{\partial E}{\partial \omega}$$
其中$E$代表误差函数，在这里我将总结一下目前最常用的2种误差，均方误差与交叉熵误差在误差逆传播算法中对权重的影响  
首先是交叉熵误差：  
