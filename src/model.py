import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF

"""
建立网络模型结构
"""


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        max_length = config['max_length']
        class_num = config['class_num']
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config['use_crf']
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x)  # input shape:(batch_size, sen_len)
        x, _ = self.layer(x)  # input shape:(batch_size, sen_len, input_dim)
        predict = self.classify(x)  # input shape:(batch_size, input_dim)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction='token_mean')
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from config import config

    config['class_num'] = 20
    config['vocab_size'] = 20
    config['max_length'] = 5

    model = Model(config)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    label = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    print(model(x, label))
