import torch
import re
import numpy as np
from collections import defaultdict
from src.loader import load_dataset


def decode(sentence, labels):
    """{
        "B_LOC": 0,
        "B_ORG": 1,
        "B_PER": 2,
        "B_T": 3,
        "I_LOC": 4,
        "I_ORG": 5,
        "I_PER": 6,
        "I_T": 7,
        "O": 8
    }"""
    labels = "".join([str(x) for x in labels[:len(sentence)]])
    results = defaultdict(list)
    for location in re.finditer("(04+)", labels):
        s, e = location.span()
        results["LOC"].append(sentence[s:e])
    for location in re.finditer("(15+)", labels):
        s, e = location.span()
        results["ORG"].append(sentence[s:e])
    for location in re.finditer("(26+)", labels):
        s, e = location.span()
        results["PER"].append(sentence[s:e])
    for location in re.finditer("(37+)", labels):
        s, e = location.span()
        results["T"].append(sentence[s:e])
    return results


class EvalData:
    def __init__(self, model, config, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.config['train'] = False
        self.test_data = load_dataset(config, shuffle=False)

    def eval(self, epoch):
        self.logger.info('测试第%d轮模型效果:' % epoch)
        self.pre_dict = {'LOC': defaultdict(int),
                         'ORG': defaultdict(int),
                         'PER': defaultdict(int),
                         'T': defaultdict(int)}
        self.model.eval()
        for index, data in enumerate(self.test_data):
            sentences = self.test_data.dataset.sentences[index * self.config['batch_size']:
                                                         (index + 1) * self.config['batch_size']]
            if torch.cuda.is_available():
                data = [d.cuda() for d in data]
            input_ids, labels = data
            with torch.no_grad():
                pred = self.model(input_ids)  # 不输入labels，只用模型来预测
            self.get_result(labels, pred, sentences)
        self.show_result()
        return self.pre_dict

    def get_result(self, labels, pred, sentences):
        assert len(labels) == len(pred) == len(sentences)
        if not self.config['use_crf']:
            pred = torch.argmax(pred, dim=-1)
        for true_label, pred_label, sentece in zip(labels, pred, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entites = decode(sentece, true_label)
            pred_entites = decode(sentece, pred_label)
            for key in ["LOC", "ORG", "PER", "T"]:
                self.pre_dict[key]['正确识别数'] += len([entity for entity in pred_entites[key] if entity in true_entites[key]])
                self.pre_dict[key]['样本实体数'] += len(true_entites[key])
                self.pre_dict[key]['识别出实体数'] += len(pred_entites[key])

    def show_result(self):
        F1_scores = []
        for key in ["LOC", "ORG", "PER", "T"]:
            precision = self.pre_dict[key]['正确识别数'] / (1e-5 + self.pre_dict[key]['识别出实体数'])
            recall = self.pre_dict[key]['正确识别数'] / (1e-5 + self.pre_dict[key]['样本实体数'])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info('%s类实体，准确率：%f，召回率：%f，F1：%f' % (key, precision, recall, F1))
        self.logger.info("Macro-F1：%f" % np.mean(F1_scores))
        correct_pred = sum([self.pre_dict[key]['正确识别数'] for key in ["LOC", "ORG", "PER", "T"]])
        total_pred = sum([self.pre_dict[key]['识别出实体数'] for key in ["LOC", "ORG", "PER", "T"]])
        total_true = sum([self.pre_dict[key]['样本实体数'] for key in ["LOC", "ORG", "PER", "T"]])
        micro_precision = correct_pred / (1e-5 + total_pred)
        micro_recall = correct_pred / (1e-5 + total_true)
        micro_f1 = (2 * micro_precision * micro_recall) / (1e-5 + micro_precision + micro_recall)
        self.logger.info('Micro-F1 %f' % micro_f1)
        self.logger.info('-----------------------------')