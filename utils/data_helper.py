import os
import json
import logging
import pickle
import numpy as np
import torch
from torch.utils import data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class pair_dataset(data.Dataset):

    def __init__(self, input_file, tokenizer, max_seq_len, shuffle=True):
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.data = self.load_data(input_file)

    def load_data(self, input_file):
        cache_file = input_file.replace('.json', '_cache.pkl')
        if os.path.exists(cache_file):
            logger.info("loading data from cache file: %s" % cache_file)
            return pickle.load(open(cache_file, 'rb'))
        else:
            logger.info("loading data from input file: %s" % input_file)
            with open(input_file, 'r', encoding='utf8') as f:
                i = 1
                data = []
                for line in f:
                    if i % 10000 == 0:
                        logger.info("%d examples have been loaded" % i)
                    line = json.loads(line.strip())

                    qid = line['qid']
                    desc = line["sentence"]
                    word = line["word"]
                    label = line["label"]

                    # 模型输入
                    input_ids, segment_ids = self.tokenize_sentence(desc, word)

                    label = float(label)
                    data.append(
                        {
                            'qid': qid,
                            'input_ids': input_ids,
                            'segment_ids': segment_ids,
                            'label': label,
                        }
                    )

                    i += 1

            if self.shuffle:
                np.random.shuffle(data)
            pickle.dump(data, open(cache_file, 'wb'))
            return data

    def tokenize_sentence(self, text_a, text_b):
        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_len - 3)

        tokens = [self.cls_token] + tokens_a + [self.sep_token] + tokens_b + [self.sep_token]
        segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids, segment_ids

    def __getitem__(self, idx):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


def pad_seq(insts,
            return_seq_type=False,
            return_seq_pos=False,
            return_seq_mask=False
            ):
    return_list = []

    max_len = max(len(inst) for inst in insts)

    # input ids
    inst_data = np.array(
        [inst + list([0] * (max_len - len(inst))) for inst in insts],
    )
    return_list += [inst_data.astype("int64")]

    if return_seq_type:
        # input sentence type
        return_list += [np.zeros_like(inst_data).astype("int64")]

    if return_seq_mask:
        # input mask
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        return_list += [input_mask_data.astype("float32")]

    if return_seq_pos:
        # input position
        inst_pos = np.array([list(range(0, len(inst))) + [0] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_pos.astype("int64")]

    return return_list


def collate_fn(data):
    examples = data

    return_list = []

    # input
    batch_input_ids = []
    batch_segment_ids = []
    batch_label = []
    batch_qid = []

    for i in range(len(examples)):
        example = examples[i]

        batch_input_ids.append(example['input_ids'])
        batch_segment_ids.append(example['segment_ids'])
        batch_label.append(example['label'])
        batch_qid.append(example['qid'])

    # seq pad
    return_list += pad_seq(batch_input_ids, False, True, True)  # 3
    return_list += pad_seq(batch_segment_ids)
    return_list += [batch_label]
    return_list = [torch.tensor(batch_data) for batch_data in return_list]
    return_list += [batch_qid]
    return return_list
