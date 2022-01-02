import torch
import json
from torch.utils.data import DataLoader

"""
数据加载脚本
"""

def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()  # 去除行尾换行符
            vocab_dict[token] = index + 1  # 还有padding的位置，让出0来
    return vocab_dict


def load_schema(path):
    with open(path, encoding='utf8') as f:
        return json.load(f)


class DataGenerator:
    '''
    数据生成类：文本对->longtensor数据对
    '''

    def __init__(self, config):
        self.config = config
        if config['train']:
            self.path = config['train_path']
        else:
            self.path = config['test_path']
        self.vocab = load_vocab(config['vocab_path'])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = load_schema(config["schema_path"])
        self.config['class_num'] = len(self.schema)
        self.load_data()

    def load_data(self):
        self.data = []
        self.source = open(self.path + 'source.txt', 'r', encoding='utf8').readlines()
        self.target = open(self.path + 'target.txt', 'r', encoding='utf8').readlines()
        for i in range(len(self.source)):
            self.sentence = self.source[i].replace('\n', '').split(' ')
            self.sentences.append(''.join(self.sentence))
            self.labels = [self.schema[label] for label in self.target[i].replace('\n', '').split(' ')]
            input_ids = self.encode_sentence(self.sentence)
            labels = self.padding(self.labels, -1)
            self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

    def encode_sentence(self, sentence):
        input_ids = []
        for char in sentence:
            input_ids.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_ids = self.padding(input_ids)
        return input_ids

    def padding(self, input_ids, pad_token=0):
        input_ids = input_ids[:self.config['max_length']]
        input_ids += [pad_token] * (self.config['max_length'] - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用DataLoader类封装数据
def load_dataset(config, shuffle=True):
    dg = DataGenerator(config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from config import config

    config['vocab_path'] = '../data/vocab.txt'
    config['train_path'] = '../data/train/'
    config['test_path'] = '../data/test/'
    config['schema_path'] = '../data/schema.json'
    config['train'] = False
    # 数据生成类测试样例
    # DG = DataGenerator(config)

    # 检查数据样式
    dl = load_dataset(config)
    for train_x, train_y in dl:
        break
    print(len(dl))
    print(train_x.shape, train_y.shape)
