import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from transformers import AutoModelForCausalLM, AutoTokenizer

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, output_size,activation=False,init_method='xavier_uniform',bias=False):
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
            if bias:
                self.fc.bias.data.fill_(0)  # 初始化偏置为0
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
            if bias:
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
        self.input_gate = FeedForwardNetwork(input_size + hidden_size, hidden_size)
        self.update_gate = FeedForwardNetwork(input_size + hidden_size, hidden_size)
        self.reset_gate = FeedForwardNetwork(input_size + hidden_size, hidden_size)

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


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (k, 768)) for k in [2,3,4]])
        self.dropout = nn.Dropout(0.5)
        self.fc = FeedForwardNetwork(256 * 3, 7)
        self.mish = Mish()



    def conv_and_pool(self, x, conv):
        x = self.mish(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class PaperClassifier(nn.Module):
    def __init__(self, model,tokenizer,n_classes):
        super(PaperClassifier, self).__init__()
        self.tokenizer = tokenizer

        self.model = model
        self.dims = 512
        self.new_lm_head = FeedForwardNetwork(2048,self.dims).bfloat16()

        self.dropout = nn.Dropout(0.5)

        # 替换大模型中最后一层
        self.replace_layer(self.model,"lm_head",self.new_lm_head)


        for name, param in self.model.named_parameters():
            if name == "transformer.wte.weight":
                param.requires_grad_(True)
            elif "lora" in name:
                param.requires_grad_(True)
            elif name == "lm_head.fc.weight":
                param.requires_grad_(True)
            else:
                param.requires_grad = False

        # 学习一个状态转移矩阵
        # self.markov = FeedForwardNetwork(768, 768).bfloat16()

        # 初始化的hidden_states
        # self.hc1 = FeedForwardNetwork(self.dims, self.dims).bfloat16()

        # 保持长距离的语义传输
        self.gru = GRUCell(self.dims, self.dims).bfloat16()

        # 输出层
        self.context_layer = FeedForwardNetwork(self.dims, n_classes).bfloat16()

        

    def replace_layer(self,model,target_layer,new_layer):
        """
        在模型中递归地查找并替换指定名称的层。
        
        :param model: 要修改的模型实例。
        :param target_layer: 要替换的层的名称。
        :param new_layer: 用于替换的新层实例。
        """
        for name, module in model.named_children():
            if name == target_layer:
                setattr(model, name, new_layer)
                print(f"Layer {target_layer} has been replaced.")
                return
            elif isinstance(module, nn.Module):
                self.replace_layer(module, target_layer, new_layer)
        return model


    def forward(self, input_ids, attention_mask,logits=False):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = output.logits

        # 语义提取
        start_token = output[:,0,:].unsqueeze(1)

        hidden_states = torch.zeros_like(start_token)

        for i in range(output.shape[1]):
            cur_token = output[:,i,:].unsqueeze(1)
            # input_features = cur_token
            # input_features = self.markov(cur_token)
            # input_features = self.dropout(input_features)
            hidden_states = self.gru(cur_token,hidden_states)
            # hidden_states = self.dropout(hidden_states)
        

        hidden_states = hidden_states.squeeze(1)

        # 分类层
        output = self.context_layer(hidden_states)
        output = self.dropout(output)
 
        return output

