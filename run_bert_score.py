import argparse
import logging
import os
import random

import numpy as np
import torch

from transformers import BertConfig, BertTokenizer

from utils import data_helper, train_helper

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # 1. 训练和测试数据路径
    parser.add_argument("--train_set", default='./train_data/train.json', type=str, help="Path to training data.")
    parser.add_argument("--eval_set", default='./train_data/dev.json', type=str, help="Path to eval data.")
    parser.add_argument("--test_set", default='./train_data/predict.json', type=str, help="Path to test data.")

    # 2. 预训练模型路径
    parser.add_argument("--vocab_file", default="./pretrain/vocab.txt", type=str, help="Init vocab to resume training from.")
    parser.add_argument("--config_path", default="./pretrain/config.json", type=str, help="Init config to resume training from.")
    parser.add_argument("--init_checkpoint", default="./pretrain/pytorch_model.bin", type=str, help="Init checkpoint to resume training from.")

    # 3. 保存模型
    parser.add_argument("--save_path", default="./check_points", type=str, help="Path to save checkpoints.")
    parser.add_argument("--load_path", default="./check_points/model_best.bin", type=str, help="Path to load checkpoints.")

    # 训练和测试参数
    parser.add_argument("--do_train", default=False, type=bool, help="Whether to perform training.")
    parser.add_argument("--do_eval", default=False, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_predict", default=False, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_adv", default=False, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_pseudo", default=False, type=bool)
    parser.add_argument("--do_mix", default=False, type=bool)

    parser.add_argument("--epochs", default=30, type=int, help="Number of epoches for fine-tuning.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total examples' number in batch for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total examples' number in batch for eval.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="Number of words of the longest seqence.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate used to train with warmup.")
    parser.add_argument("--warmup_proportion", default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")

    # pseudo label 参数
    parser.add_argument("--unlabeled_set", default='./train_data/unlabeled_data_15.json', type=str)
    parser.add_argument("--unlabeled_train_batch_size", default=16, type=int)
    parser.add_argument("--T1", default=200, type=int)
    parser.add_argument("--T2", default=500, type=int)
    parser.add_argument("--af", default=0.1, type=float)

    # mixup 参数
    parser.add_argument("--alpha", default=2, type=float)
    parser.add_argument('--mix-layers-set', nargs='+',
                        default=[7, 9, 12], type=int, help='define mix layer set')

    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda")
    parser.add_argument("--log_steps",
                        type=int,
                        default=20,
                        help="The steps interval to print loss.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.use_cuda:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_path_postfix = '_'
    if args.do_pseudo:
        model_path_postfix += 'pseudo_'
    if args.do_adv:
        model_path_postfix += 'adv_'
    if args.do_mix:
        model_path_postfix += 'mix_'

    args.save_path = os.path.join(args.save_path, args.train_set.split('/')[-1].split('.')[0]+model_path_postfix)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    bert_tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
    bert_config = BertConfig.from_pretrained(args.config_path)

    # 获取数据
    train_dataset = None
    eval_dataset = None
    if args.do_train:
        logger.info("loading train dataset")
        train_dataset = data_helper.pair_dataset(args.train_set, bert_tokenizer,
                                                 max_seq_len=args.max_seq_len,)

    if args.do_pseudo:
        logger.info("loading unlabeled dataset")
        unlabeled_dataset = data_helper.pair_dataset(args.unlabeled_set, bert_tokenizer,
                                                     max_seq_len=args.max_seq_len)

    if args.do_eval:
        logger.info("loading eval dataset")
        eval_dataset = data_helper.pair_dataset(args.eval_set, bert_tokenizer,
                                                max_seq_len=args.max_seq_len,
                                                shuffle=False)

    if args.do_predict:
        logger.info("loading test dataset")
        test_dataset = data_helper.pair_dataset(args.test_set, bert_tokenizer,
                                                max_seq_len=args.max_seq_len,
                                                shuffle=False)

    if args.do_train:
        if args.do_mix:
            train_helper.Mixup_train(bert_tokenizer, bert_config, args, train_dataset, eval_dataset)
        elif args.do_pseudo:
            train_helper.pseudo_train(bert_tokenizer, bert_config, args, train_dataset, unlabeled_dataset, eval_dataset)
        else:
            train_helper.train(bert_tokenizer, bert_config, args, train_dataset, eval_dataset)

    if args.do_predict:
        probs, preds, qid_to_preds = train_helper.test(bert_tokenizer, bert_config, args, test_dataset)
        # train_helper.get_pseudo_label(probs, args.test_set, './train_data/pl_data.json')


if __name__ == "__main__":
    main()
