import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        # 定义门控和线性转换层
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # 激活函数为sigmoid和tanh
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_, hidden_state):
        combined = torch.cat((input_, hidden_state), dim=2)  # torch.Size([2, 512, 4096])

        # 计算更新门和重置门信号
        z = self.sigmoid(self.update_gate(combined))
        r = self.sigmoid(self.reset_gate(combined))
        # 计算候选隐藏状态
        h_prime = self.tanh(self.input_gate(torch.cat((input_, r * hidden_state), dim=2)))
        # 更新隐藏状态
        hidden_state = (1 - z) * hidden_state + z * h_prime
        return hidden_state

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 学习一个状态转移矩阵
        self.markov = nn.Linear(768, 1024)
        # 初始化的hidden_states
        self.hc1 = nn.Linear(768, 1024)
        # 保持长距离的语义传输
        self.gru = GRUCell(1024,1024)

    def forward(self,x):
        start_token = x[:,0,:]
        hidden_states = self.hc1(start_token).unsqueeze(1)
        for i in range(x.shape[1]):
            cur_token = x[:,i,:].unsqueeze(1)
            input_features = self.markov(cur_token)
            hidden_states = self.gru(input_features,hidden_states)
        return hidden_states