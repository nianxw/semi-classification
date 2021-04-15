import os
import json
import logging
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from models import bert_for_score, fgm, bert_for_mixtext
from utils import util
from utils import data_helper
from utils import metric

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def data_to_cuda(batch):
    return_lists = []
    for t in batch:
        if isinstance(t, torch.Tensor):
            return_lists += [t.cuda()]
        else:
            return_lists += [t]
    return return_lists


def batch_forward(batch, model, loss_fc):
    pair_labels = batch[4]
    pair_probs, pair_preds = model(batch)
    loss = loss_fc(pair_probs, pair_labels)
    return loss, pair_preds


def batch_forward_mix(input_a, target, model,
                      input_b=None,
                      lambda_l=None,
                      mix_layer=None):
    logits = model(input_a, input_b, lambda_l, mix_layer)
    _, preds = torch.max(F.softmax(logits, dim=-1), dim=-1)
    loss = - torch.mean(torch.sum(F.log_softmax(logits, dim=1) * target, dim=1))
    return loss, preds


def train(tokenizer, config, args, train_data_set, eval_data_set=None):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   #    num_workers=8,
                                   collate_fn=data_helper.collate_fn)
    if args.do_eval:
        eval_data_loader = DataLoader(dataset=eval_data_set,
                                      batch_size=args.eval_batch_size,
                                      #   num_workers=8,
                                      collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    model = bert_for_score.BertScore.from_pretrained(
        args.init_checkpoint, config=config)
    if args.do_adv:
        fgm_model = fgm.FGM(model)  # 定义对抗训练模型

    if args.use_cuda:
        # model = nn.DataParallel(model)
        model.cuda()

    if args.load_path is not None and os.path.exists(args.load_path):
        ckpt = args.load_path
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        logger.info("Successfully loaded checkpoint '%s'" % ckpt)

    # prepare optimizer
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
        {'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters_to_optimize,
                      lr=args.learning_rate, correct_bias=False)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()

    log_loss = 0.0
    log_pair_preds = None
    log_pair_labels = None

    best_f1 = 0.0

    pair_loss_fc = nn.BCELoss()
    begin_time = time.time()
    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)
            loss, pair_preds = batch_forward(batch, model, pair_loss_fc)
            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                loss_adv, _ = batch_forward(batch, model, pair_loss_fc)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm_model.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log_loss += loss.data.item()
            pair_preds = pair_preds.view(-1).data.cpu().numpy()
            pair_labels = batch[4].view(-1).cpu().numpy()

            if log_pair_preds is None:
                log_pair_preds = pair_preds
            else:
                log_pair_preds = np.concatenate(
                    (log_pair_preds, pair_preds), axis=0)

            if log_pair_labels is None:
                log_pair_labels = pair_labels
            else:
                log_pair_labels = np.concatenate(
                    (log_pair_labels, pair_labels), axis=0)

            if steps % args.log_steps == 0:
                end_time = time.time()
                used_time = end_time - begin_time
                log_f1 = metric.get_metric('f1_macro')(
                    log_pair_preds, log_pair_labels)
                logger.info(
                    "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                    "ave f1: %f, speed: %f s/step" %
                    (
                        epoch, steps, num_train_optimization_steps, steps,
                        log_loss / args.log_steps, log_f1,
                        used_time / args.log_steps,
                    ),
                )
                begin_time = time.time()
                log_loss = 0.0
                log_pair_preds = None
                log_pair_labels = None

        if args.do_eval:
            eval_res = evaluate(args, eval_data_loader, model, pair_loss_fc, epoch)
            if eval_res > best_f1:
                logging.info('save model: %s' % os.path.join(
                    args.save_path, 'model_%d.bin' % epoch))
                if isinstance(model, nn.DataParallel):
                    save_model = model.module
                else:
                    save_model = model
                torch.save({'state_dict': save_model.state_dict()}, os.path.join(
                    args.save_path, 'model_best.bin'))
                best_f1 = eval_res
                logging.info('best f1: %.4f' % best_f1)


def unlabeled_weight(step, T1, T2, af):
    alpha = 0.0
    if step > T1:
        alpha = (step - T1) / (T2 - T1) * af
        if step > T2:
            alpha = af
    return alpha


# 采用伪标签进行训练
def pseudo_train(tokenizer, config, args,
                 train_data_set,
                 unlabeled_data_set,
                 eval_data_set=None):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   #    num_workers=8,
                                   collate_fn=data_helper.collate_fn)
    unlabeled_data_loader = DataLoader(dataset=unlabeled_data_set,
                                       batch_size=args.train_batch_size,
                                       #  num_workers=8,
                                       collate_fn=data_helper.collate_fn)
    if args.do_eval:
        eval_data_loader = DataLoader(dataset=eval_data_set,
                                      batch_size=args.eval_batch_size,
                                      #   num_workers=8,
                                      collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    model = bert_for_score.BertScore.from_pretrained(
        args.init_checkpoint, config=config)
    if args.do_adv:
        fgm_model = fgm.FGM(model)

    if args.use_cuda:
        # model = nn.DataParallel(model)
        model.cuda()

    if args.load_path is not None and os.path.exists(args.load_path):
        ckpt = args.load_path
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        logger.info("Successfully loaded checkpoint '%s'" % ckpt)

    # prepare optimizer
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
        {'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters_to_optimize,
                      lr=args.learning_rate, correct_bias=False)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()

    log_loss = 0.0
    log_pair_preds = None
    log_pair_labels = None

    best_f1 = 0.0

    pair_loss_fc1 = nn.BCELoss()
    pair_loss_fc2 = nn.BCELoss(reduce=False)
    begin_time = time.time()

    unlabeled_data_iter = iter(unlabeled_data_loader)
    for epoch in range(args.epochs):
        for batch in train_data_loader:
            try:
                unlabel_batch = next(unlabeled_data_iter)
            except:
                unlabeled_data_iter = iter(unlabeled_data_loader)
                unlabel_batch = next(unlabeled_data_iter)
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)
                unlabel_batch = data_to_cuda(unlabel_batch)
            labeled_loss, pair_preds = batch_forward(batch, model, pair_loss_fc1)
            loss = labeled_loss

            model.eval()
            unlabeled_probs, unlabeled_preds = model(unlabel_batch)

            # 卡阈值
            # 求解mask矩阵

            mask_for_one = (unlabeled_probs > 0.8).long()
            mask_for_zero = (unlabeled_probs < 0.01).long()
            mask = (mask_for_one + mask_for_zero).detach().float()

            pseudo_labels = unlabeled_preds.detach().float()
            unlabeled_loss = pair_loss_fc2(unlabeled_probs, pseudo_labels)
            unlabeled_loss = torch.sum(unlabeled_loss*mask)

            loss = labeled_loss + unlabeled_weight(steps, args.T1, args.T2, args.af)*unlabeled_loss

            model.train()
            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                loss_adv_anotate, _ = batch_forward(batch, model, pair_loss_fc1)
                loss_adv_anotate.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                unlabeled_probs_adv, _ = model(unlabel_batch)
                loss_adv_unlabeled = pair_loss_fc2(unlabeled_probs_adv, pseudo_labels)
                loss_adv_unlabeled = torch.sum(loss_adv_unlabeled*mask)
                loss_adv_unlabeled.backward()
                fgm_model.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log_loss += loss.data.item()
            pair_preds = pair_preds.view(-1).data.cpu().numpy()
            pair_labels = batch[4].view(-1).cpu().numpy()

            if log_pair_preds is None:
                log_pair_preds = pair_preds
            else:
                log_pair_preds = np.concatenate(
                    (log_pair_preds, pair_preds), axis=0)

            if log_pair_labels is None:
                log_pair_labels = pair_labels
            else:
                log_pair_labels = np.concatenate(
                    (log_pair_labels, pair_labels), axis=0)

            if steps % args.log_steps == 0:
                end_time = time.time()
                used_time = end_time - begin_time
                log_f1 = metric.get_metric('f1_macro')(
                    log_pair_preds, log_pair_labels)
                logger.info(
                    "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                    "ave f1: %f, speed: %f s/step" %
                    (
                        epoch, steps, num_train_optimization_steps, steps,
                        log_loss / args.log_steps, log_f1,
                        used_time / args.log_steps,
                    ),
                )
                begin_time = time.time()
                log_loss = 0.0
                log_pair_preds = None
                log_pair_labels = None

        if args.do_eval:
            eval_res = evaluate(args, eval_data_loader, model, pair_loss_fc1, epoch)
            if eval_res > best_f1:
                logging.info('save model: %s' % os.path.join(
                    args.save_path, 'model_%d.bin' % epoch))
                if isinstance(model, nn.DataParallel):
                    save_model = model.module
                else:
                    save_model = model
                torch.save({'state_dict': save_model.state_dict()}, os.path.join(
                    args.save_path, 'model_best.bin'))
                best_f1 = eval_res
                logging.info('best f1: %.4f' % best_f1)


# Mixup 辅助训练
def Mixup_train(tokenizer, config, args,
                train_data_set,
                eval_data_set=None):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   #    num_workers=8,
                                   collate_fn=data_helper.collate_fn)
    if args.do_eval:
        eval_data_loader = DataLoader(dataset=eval_data_set,
                                      batch_size=args.eval_batch_size,
                                      #   num_workers=8,
                                      collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    model = bert_for_mixtext.MixText()
    if args.do_adv:
        fgm_model = fgm.FGM(model)

    if args.use_cuda:
        # model = nn.DataParallel(model)
        model.cuda()

    if args.load_path is not None and os.path.exists(args.load_path):
        ckpt = args.load_path
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        logger.info("Successfully loaded checkpoint '%s'" % ckpt)

    # prepare optimizer
    parameters_to_optimize = [
            {"params": model.bert.parameters(), "lr": args.learning_rate},
            {"params": model.linear.parameters(), "lr": 0.001},
    ]
    optimizer = AdamW(parameters_to_optimize)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()

    log_loss = 0.0
    best_f1 = 0.0

    pair_loss_fc = nn.CrossEntropyLoss()

    begin_time = time.time()

    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)

            input_a = batch[0]
            batch_size = input_a.shape[0]
            pair_idx = torch.randperm(batch_size)
            input_b = input_a[pair_idx]

            target_a = batch[4].long()
            target_a = torch.zeros(batch_size, 2, device=target_a.device).scatter_(1, target_a.view(-1, 1), 1)
            target_b = target_a[pair_idx]

            lambda_l = np.random.beta(args.alpha, args.alpha)
            mixed_target = lambda_l * target_a + (1 - lambda_l) * target_b

            mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
            mix_layer = mix_layer - 1

            loss, _ = batch_forward_mix(input_a, mixed_target, model, input_b, lambda_l, mix_layer)
            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                logits_adv = batch_forward_mix(input_a, mixed_target, model, input_b, lambda_l, mix_layer)[0]
                logits_adv.backward()
                fgm_model.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log_loss += loss.data.item()

            if steps % args.log_steps == 0:
                end_time = time.time()
                used_time = end_time - begin_time
                logger.info(
                    "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                    "speed: %f s/step" %
                    (
                        epoch, steps, num_train_optimization_steps, steps,
                        log_loss / args.log_steps,
                        used_time / args.log_steps,
                    ),
                )
                begin_time = time.time()
                log_loss = 0.0

        if args.do_eval:
            eval_res = evaluate(args, eval_data_loader, model, pair_loss_fc, epoch)
            if eval_res > best_f1:
                logging.info('save model: %s' % os.path.join(
                    args.save_path, 'model_%d.bin' % epoch))
                if isinstance(model, nn.DataParallel):
                    save_model = model.module
                else:
                    save_model = model
                torch.save({'state_dict': save_model.state_dict()}, os.path.join(
                    args.save_path, 'model_best.bin'))
                best_f1 = eval_res
                logging.info('best f1: %.4f' % best_f1)


def evaluate(args, eval_data_loader, model, pair_loss_fc, epoch):
    model.eval()
    with torch.no_grad():
        eval_total_loss = 0.0
        eval_total_pair_preds = None
        eval_total_pair_labels = None

        eval_steps = 0
        eval_begin_time = time.time()
        for batch_eval in eval_data_loader:
            eval_steps += 1
            if args.use_cuda:
                batch_eval = data_to_cuda(batch_eval)
            input_a = batch_eval[0]
            target_a = batch_eval[4].long()
            target_a = torch.zeros(input_a.shape[0], 2, device=target_a.device).scatter_(1, target_a.view(-1, 1), 1)
            if not args.do_mix:
                eval_loss, eval_pair_preds = batch_forward(
                    batch_eval, model, pair_loss_fc)
            else:
                eval_loss, eval_pair_preds = batch_forward_mix(
                    input_a, target_a, model)

            eval_total_loss += eval_loss.data.item()
            eval_pair_preds = eval_pair_preds.data.view(
                -1).cpu().numpy()
            eval_pair_labels = batch_eval[4].data.view(
                -1).cpu().numpy()

            if eval_total_pair_preds is None:
                eval_total_pair_preds = eval_pair_preds
            else:
                eval_total_pair_preds = np.concatenate(
                    (eval_total_pair_preds, eval_pair_preds), axis=0)

            if eval_total_pair_labels is None:
                eval_total_pair_labels = eval_pair_labels
            else:
                eval_total_pair_labels = np.concatenate(
                    (eval_total_pair_labels, eval_pair_labels), axis=0)

        eval_res = metric.get_metric('f1_macro')(
            eval_total_pair_preds, eval_total_pair_labels)
        eval_end_time = time.time()
        logger.info("***** Running evaluating *****\n")
        logger.info(
            "eval result —— epoch: %d, "
            "loss: %f, "
            "f1: %f, "
            "used time: %.6f " %
            (
                epoch,  eval_total_loss / eval_steps,
                eval_res,
                eval_end_time - eval_begin_time,
            ),
        )
        logger.info("*****************************")
    model.train()
    return eval_res


def test(tokenizer, config, args, dataset):
    num_test_steps = int(len(dataset) / args.eval_batch_size)
    model = bert_for_score.BertScore(config=config)
    if args.load_path is not None and os.path.exists(args.load_path):
        ckpt = args.load_path
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        logger.info("Successfully loaded checkpoint '%s'" % ckpt)
    if args.use_cuda:
        # model = nn.DataParallel(model)
        model.cuda()
    test_data_loader = DataLoader(dataset=dataset,
                                  batch_size=args.eval_batch_size,
                                #   num_workers=8,
                                  collate_fn=data_helper.collate_fn)

    logger.info("***** Running predicting *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Num steps = %d", num_test_steps)

    model.eval()
    qid_total = []
    preds_total = []
    probs_total = []
    for batch in test_data_loader:
        if args.use_cuda:
            batch = data_to_cuda(batch)
        qid_total.extend(batch[-1])
        pair_probs, pair_preds = model(batch)
        pair_probs = pair_probs.data.view(-1).cpu().tolist()
        probs_total.extend(pair_probs)  # 记录概率值
        pair_preds = pair_preds.data.view(-1).cpu().tolist()  # [batch_size]
        preds_total.extend(pair_preds)

    qid_to_pred = defaultdict(list)
    for qid, pred in zip(qid_total, preds_total):
        qid_to_pred[qid].append(pred)

    return probs_total, preds_total, qid_to_pred


def get_pseudo_label(probs, test_path, new_data_path):
    # 预测标签
    out = open(new_data_path, 'w', encoding='utf8')
    count_pos = 0
    count_neg = 0
    with open(test_path, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            line = json.loads(line)
            if probs[i] >= 0.9:
                line['label'] = 1
                count_pos += 1
                out.write(json.dumps(line, ensure_ascii=False)+'\n')
            elif probs[i] <= 0.002:
                line['label'] = 0
                count_neg += 1
                if count_neg <= 1000:
                    out.write(json.dumps(line, ensure_ascii=False)+'\n')
            i += 1
    logger.info('predicted | postive data: %d | negtive data: %d' %(count_pos, count_neg))
    out.close()


