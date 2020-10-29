import json
import torch
import numpy as np
import os
import time

import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from DataProcessing.data_batchiter import create_tra_batch, create_eval_batch
from Model.loss import *
from Model.vat import VATLoss


def train(bert_model, domain_model, usual_tra_data_set, virus_tra_data_set, usual_dev_data_set, virus_dev_data_set,
          usual_vat_data, virus_vat_data, tag_vocab, config, domain_vocab, tokenizer, usual_eval_data, virus_eval_data):

    usual_tra_data_set.extend(virus_tra_data_set)
    usual_vat_data.extend(virus_vat_data)

    batch_num = int(np.ceil(len(usual_tra_data_set) / float(config.tra_batch_size)))

    bert_no_decay = ['bias', 'LayerNorm.weight']
    bert_optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in bert_no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in bert_no_decay)], 'weight_decay': 0.0}
    ]

    domain_no_decay = ['bias', 'LayerNorm.weight']
    domain_optimizer_grouped_parameters = [
            {'params': [p for n, p in domain_model.named_parameters() if not any(nd in n for nd in domain_no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in domain_model.named_parameters() if any(nd in n for nd in domain_no_decay)],
             'weight_decay': 0.0}
        ]

    domain_optimizer = AdamW(domain_optimizer_grouped_parameters, lr=config.domain_lr, eps=config.epsilon)
    domain_scheduler = get_linear_schedule_with_warmup(domain_optimizer, num_warmup_steps=0, num_training_steps=config.epoch * batch_num)

    optimizer_bert = AdamW(bert_optimizer_grouped_parameters, lr=config.bert_lr, eps=config.epsilon)
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=config.epoch * batch_num)


    # Get start!
    global_step = 0

    best_tra_f1 = 0
    best_cla_acc = 0
    best_domain_acc = 0
    best_dev_usual = 0
    best_dev_virus = 0

    critierion = LabelSmoothing(config)

    vat_loss = VATLoss(config)

    for epoch in range(0, config.epoch):
        gold_label = []
        predict_ids = []
        cla_score = 0
        domain_score = 0
        print('\nThe epoch is starting.')
        epoch_start_time = time.time()
        batch_iter = 0
        print('The epoch is :', str(epoch))
        for all_batch in create_tra_batch(usual_tra_data_set, usual_vat_data, tag_vocab, config.tra_batch_size, config,
                                          tokenizer, domain_vocab, shuffle=True):
            start_time = time.time()
            word_batch = all_batch[0]
            vat_word_batch = all_batch[1]
            bert_model.train()
            domain_model.train()

            batch_size = word_batch[0][0].size(0)
            input_tensor = word_batch[0]
            target = word_batch[1]
            domain_target = word_batch[2]
            gold_label.extend(target)

            ul_input_tensor = vat_word_batch[0]
            lds = vat_loss(bert_model, ul_input_tensor)

            logits, last_hidden = bert_model(input_tensor)
            domain_logits = domain_model(last_hidden)

            cla_loss = critierion(logits, target)
            correct, predict_id, accuracy = class_loss(logits, target)
            domain_loss, domain_correct, domain_accuracy = domain_cla_loss(domain_logits, domain_target)
            predict_ids.extend(predict_id)

            loss = (cla_loss + domain_loss + config.vat_alpha * lds) / config.update_every
            loss.backward()
            cla_loss_value = cla_loss.item()
            domain_loss_value = domain_loss.item()
            vat_loss_value = (config.vat_alpha * lds).item()
            during_time = float(time.time() - start_time)
            print('Step:{}, Epoch:{}, batch_iter:{}, cla_accuracy:{:.4f}({}/{}),'
                  'domain_accuracy:{:.4f}({}/{}), time:{:.2f}, '
                  'cla_loss:{:.6f}, domain_loss:{:.6f}, vat_loss:{:.6f}'.format(global_step, epoch, batch_iter, accuracy,
                                                               correct, batch_size, domain_accuracy, domain_correct,
                                                               batch_size, during_time, cla_loss_value, domain_loss_value,
                                                               vat_loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                if config.clip_max_norm_use:
                    nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=config.clip)
                    nn.utils.clip_grad_norm_(domain_model.parameters(), max_norm=config.clip)

                optimizer_bert.step()
                domain_optimizer.step()

                scheduler_bert.step()
                domain_scheduler.step()

                bert_model.zero_grad()
                domain_model.zero_grad()

                global_step += 1
            cla_score += correct
            domain_score += domain_correct

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                print("now bert lr is {}".format(optimizer_bert.param_groups[0].get("lr")), '\n')
                dev_usual_score, weight = evaluate(bert_model, usual_dev_data_set, config, tag_vocab, domain_vocab, tokenizer)
                if best_dev_usual < dev_usual_score:
                    print('the best usual_dev score is: acc:{}'.format(dev_usual_score))
                    best_dev_usual = dev_usual_score
                    decoder(bert_model, usual_eval_data, config, tag_vocab, tokenizer, domain_vocab, weight)
                    if os.path.exists(config.save_model_path):
                        torch.save(bert_model.state_dict(), config.usual_model_pkl)
                    else:
                        os.makedirs(config.save_model_path)
                        torch.save(bert_model.state_dict(), config.usual_model_pkl)

                dev_virus_score, weight = evaluate(bert_model, virus_dev_data_set, config, tag_vocab, domain_vocab,
                                                   tokenizer, test=True)
                if best_dev_virus < dev_virus_score:
                    print('the best virus_dev score is: acc:{}'.format(dev_virus_score) + '\n')
                    best_dev_virus = dev_virus_score
                    decoder(bert_model, virus_eval_data, config, tag_vocab, tokenizer, domain_vocab, weight, test=True)
                    if os.path.exists(config.save_model_path):
                        torch.save(bert_model.state_dict(), config.virus_model_pkl)
                    else:
                        os.makedirs(config.save_model_path)
                        torch.save(bert_model.state_dict(), config.virus_model_pkl)
        epoch_time = float(time.time() - epoch_start_time)
        tra_score = get_Macro_F1_score(gold_label, predict_ids, tag_vocab)
        all_cla_score = 100.0 * cla_score / len(usual_tra_data_set)
        all_domain_score = 100.0 * domain_score / len(usual_tra_data_set)
        if tra_score > best_tra_f1:
            best_tra_f1 = tra_score
            print('the best_train F1 is:{:.2f}'.format(best_tra_f1))
        if all_cla_score > best_cla_acc:
            best_cla_acc = all_cla_score
            print('the best_train cla_score is:{}({}/{})'.format(best_cla_acc, cla_score, len(usual_tra_data_set)))
        if all_domain_score > best_domain_acc:
            best_domain_acc = all_domain_score
            print('the best_train domain_score is:{}({}/{})'.format(best_domain_acc, domain_score, len(usual_tra_data_set)))
        print("epoch_time is:", epoch_time)


def evaluate(bert_model, dev_data, config, tag_vocab, domain_vocab, tokenizer, test=False):
    bert_model.eval()
    get_score = 0
    start_time = time.time()
    all_logit = []
    gold_label = []
    predict_ids = []
    for word_batch in create_eval_batch(dev_data, tag_vocab, config.test_batch_size, config, tokenizer, domain_vocab):
        input_tensor = word_batch[0]
        target = word_batch[1]
        gold_label.extend(target)
        logits, _ = bert_model(input_tensor)
        new_logits = torch.Tensor(logits.data.tolist())
        all_logit.append(new_logits)
        correct, predict_id, accuracy = class_loss(logits, target)
        predict_ids.extend(predict_id)
        get_score += correct
    all_logit = torch.cat(all_logit, dim=0)
    optimized_f1 = F1Optimized(all_logit, gold_label, tag_vocab, config)
    optimized_f1.optimized()
    weight = optimized_f1.res
    new_f1 = optimized_f1.cau_f1()
    if test:
        dev_score = get_Macro_F1_score(gold_label, predict_ids, tag_vocab)
        print('the current_test virus_score is: F1:{:.2f}'.format(dev_score))
        print('the current_test virus_score is: New F1:{:.2f}'.format(new_f1))
    else:
        dev_score = get_Macro_F1_score(gold_label, predict_ids, tag_vocab)
        print('the current_dev usual_score is: F1:{:.2f}'.format(dev_score))
        print('the current_dev usual_score is: New F1:{:.2f}'.format(new_f1))
    during_time = float(time.time() - start_time)
    print('spent time is:{:.4f}'.format(during_time))
    return new_f1, weight


def decoder(bert_model, eval_data, config, tag_vocab, tokenizer, domain_vocab, weight, test=False):
    bert_model.eval()
    data_ids = []
    all_logit = []
    for word_batch in create_eval_batch(eval_data, tag_vocab, config.test_batch_size, config, tokenizer, domain_vocab):
        input_tensor = word_batch[0]
        data_id = word_batch[3]
        data_ids.extend(data_id)
        logits, _ = bert_model(input_tensor)
        new_logits = torch.Tensor(logits.data.tolist())
        all_logit.append(new_logits)
    if test:
        path = config.save_virus_path
    else:
        path = config.save_usual_path
    all_logit = torch.cat(all_logit, dim=0)
    all_logit = all_logit.data.numpy()
    new_logits = torch.from_numpy(all_logit * weight)
    predict_ids = smp_eval(new_logits)
    json_list = []
    for index, predict_id in zip(data_ids, predict_ids):
        submit_dic = {}
        submit_dic["id"] = index[0]
        submit_dic["label"] = tag_vocab.id2word(predict_id)
        json_list.append(submit_dic)
    json_list = sorted(json_list, key=lambda d: d['id'])
    json_str = json.dumps(json_list)
    with open(path, 'w', encoding='utf8') as f:
        f.write(json_str)
    print('Write over.')








