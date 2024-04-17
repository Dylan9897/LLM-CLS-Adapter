import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from transformers import AutoModelForCausalLM, AutoTokenizer

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, output_size,activation=False,init_method='xavier_uniform',bias=True):
        super(FeedForwardNetwork, self).__init__()
        # 定义全连接层
        if not bias:
            self.fc = nn.Linear(input_size, output_size,bias=False)
        else:
            self.fc = nn.Linear(input_size, output_size)
        
        self.activation = activation
        self.dropout = nn.Dropout(0.5)

        # 参数初始化
        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.data.fill_(0)  # 初始化偏置为0
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
            self.fc.bias.data.fill_(0)  # 初始化偏置为0
        else:
            raise ValueError('Unsupported initialization method')

    def forward(self, x):
        out = self.fc(x)
        if self.activation:
            out = self.activation(out) 
        return out

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        print('Mish activation loaded')

    def forward(self,x):
        out = F.softplus(x)
        x = x*(torch.tanh(softplus(x)))
        return x

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

class PaperClassifier(nn.Module):
    def __init__(self, model,tokenizer,n_classes):
        super(PaperClassifier, self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        for name, param in self.model.named_parameters():
            if name == "transformer.wte.weight":
                param.requires_grad_(True)
            else:
                param.requires_grad = False

        self.mish = Mish()

        # 学习一个状态转移矩阵
        self.markov = FeedForwardNetwork(2048, 2048,activation=self.mish).bfloat16()

        # 初始化的hidden_states
        self.hc1 = FeedForwardNetwork(2048, 2048,activation=self.mish).bfloat16()

        # 保持长距离的语义传输
        self.gru = GRUCell(2048, 2048).bfloat16()

        # 降维层
        self.reduce_dim = FeedForwardNetwork(151936, 2048,activation=self.mish).bfloat16()

        # 输出层
        self.context_layer = FeedForwardNetwork(2048, n_classes).bfloat16().bfloat16()

        self.drop = nn.Dropout(p=0.5)


    def forward(self, input_ids, attention_mask,logits=False):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = output.logits

        # output = output.mean(dim=1).squeeze(1)
        # 降维
        output = self.reduce_dim(output)

        # 语义提取
        start_token = output[:,0,:].unsqueeze(1)

        hidden_states = self.hc1(start_token)

        for i in range(output.shape[1]):
            cur_token = output[:,i,:].unsqueeze(1)
            input_features = self.markov(cur_token)
            hidden_states = self.gru(input_features,hidden_states)
        

        hidden_states = hidden_states.squeeze(1)

        # 分类层
        output = self.context_layer(hidden_states)
        output = self.drop(output)
        return output

