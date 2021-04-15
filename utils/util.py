import json
import torch
import numpy as np
import torch.nn.functional as F


def read_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            sentences.append(text)
            entities = []
            if 'label' in line:
                for v in line['label'].values():
                    entities.extend(list(v.keys()))
                labels.append(entities)
    return sentences, labels


class Trie(object):
    def __init__(self):
        self.root = {}

    def insert_word(self, word):
        root = self.root
        for c in word:
            if c not in root:
                root[c] = {}
            root = root[c]
        root['is_end'] = True

    def search(self, sentence):
        i = 0
        res = []
        while i < len(sentence):
            root = self.root
            while i < len(sentence) and sentence[i] not in root:
                i += 1
                j = i
            tmp = []
            while j < len(sentence) and sentence[j] in root:
                root = root[sentence[j]]
                j += 1
                if root.get('is_end', False):
                    tmp.append([i, j])
            if tmp:
                res.append(tmp[-1])
                i = tmp[-1][1]
            else:
                i += 1
        return res


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, args, mixed=1):

        Lx = - \
            torch.mean(torch.sum(F.log_softmax(
                outputs_x, dim=1) * targets_x, dim=1))

        probs_u = torch.softmax(outputs_u, dim=1)

        Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

        Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                               * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch, args.epochs), Lu2, args.lambda_u_hinge * linear_rampup(epoch, args.epochs)
