torch.nn.LSTM中:  
input_size – The number of expected features in the input x  
hidden_size – The number of features in the hidden state h  
input_size代表输入的维度，hidden_size代表LSTM的个数  
  
torch.nn.Conv1d中:  
input size (N, $C_{\text{in}}$, L)  
output (N, $C_{\text{out}}$, $L_{\text{out}}$)  
这个维度变换是因为每一个卷积核移动计算的时候，对一张图的路径上的所有值是累加成一个值
