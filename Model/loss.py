import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import Counter
from scipy import optimize


def class_loss(logit, gold):
    batch_size = logit.size(0)
    predict_id = torch.max(logit, 1)[1].view(gold.size()).data
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    return correct, predict_id, accuracy


def distinguish_loss(logit, gold):
    batch_size = logit.size(0)
    loss = F.cross_entropy(logit, gold)
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    return loss, correct, accuracy


def domain_cla_loss(logit, gold):
    batch_size = logit.size(0)
    loss = F.cross_entropy(logit, gold)
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    return loss, correct, accuracy


def smp_eval(logit):
    predict_id = torch.max(logit, 1)[1]
    return predict_id


def get_F1_score(gold_num, pred_num, correct_num):
    p = float(correct_num) / pred_num if pred_num != 0 else 0
    r = float(correct_num) / gold_num if gold_num != 0 else 0
    f1 = 200.0 * correct_num / (gold_num + pred_num) if gold_num + pred_num != 0 else 0.
    return p, r, f1


def get_Macro_F1_score(gold_label, predict_ids, tag_vocab, flag=True):

    total_f1 = 0

    gold_counter = Counter()
    pre_counter = Counter()
    correct_counter = Counter()

    if len(gold_label) != len(predict_ids):
        print('Error!!!!!!!!!')

    for gold, pre in zip(gold_label, predict_ids):
        gold_counter[int(gold)] += 1
        pre_counter[int(pre)] += 1
        if gold == pre:
            correct_counter[tag_vocab.id2word(gold)] += 1

    gold_dict = dict(gold_counter)
    pre_dict = dict(pre_counter)
    correct_dict = dict(correct_counter)

    for key in gold_dict.keys():
        if tag_vocab.id2word(key) not in correct_dict.keys():
            correct_dict[tag_vocab.id2word(key)] = 0
        if key not in pre_dict.keys():
            pre_dict[key] = 0

    if flag:
        print('--------------------evaluate------------------------')
    for key in gold_dict.keys():
        p, r, f1 = get_F1_score(gold_dict[key], pre_dict[key], correct_dict[tag_vocab.id2word(key)])
        if flag:
            print('{}:\t\tP:{:.2f}%\tR:{:.2f}%\tF1:{:.2f}%'.format(tag_vocab.id2word(key), 100 * p, 100 * r, f1))
        total_f1 += f1
    macro_f1 = total_f1 / len(gold_dict)
    return macro_f1


class LabelSmoothing(nn.Module):

    def __init__(self, config):
        super(LabelSmoothing, self).__init__()
        self.config = config
        self.LogSoftmax = nn.LogSoftmax()

        if self.config.label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False)
        self.confidence = 1.0 - self.config.label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.config.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, logits, labels):
        scores = self.LogSoftmax(logits)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        if self.confidence < 1:
            tdata = labels.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if self.config.use_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(labels.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            labels = tmp_.detach()
        loss = self.criterion(scores, labels)
        return loss


class F1Optimized(object):

    def __init__(self, logits, target, tag_vocab, config):
        self.config = config
        self.weight = np.random.randn(config.class_num)
        self.logits = logits.data.numpy()
        self.target = target
        self.tag_vocab = tag_vocab
        self.res = None

    def fun(self, x):
        new_logits = np.hstack(
            [x[0] * self.logits[:, 0].reshape(-1, 1), x[1] * self.logits[:, 1].reshape(-1, 1),
             x[2] * self.logits[:, 2].reshape(-1, 1), x[3] * self.logits[:, 3].reshape(-1, 1),
             x[4] * self.logits[:, 4].reshape(-1, 1), x[5] * self.logits[:, 5].reshape(-1, 1)])
        new_logits = torch.from_numpy(new_logits)
        predict_id = smp_eval(new_logits)
        predict_id = torch.split(predict_id, 1)
        return -get_Macro_F1_score(self.target, predict_id, self.tag_vocab, flag=False)

    def optimized(self):
        self.res = optimize.fmin_powell(self.fun, self.weight, disp=False)

    def cau_f1(self):
        new_logits = torch.from_numpy(self.logits * self.res)
        predict_id = smp_eval(new_logits)
        predict_id = torch.split(predict_id, 1)
        return get_Macro_F1_score(self.target, predict_id, self.tag_vocab, flag=False)












