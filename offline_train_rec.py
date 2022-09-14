import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
import random
from tqdm import tqdm
import datetime
import json
import math
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import argparse
from recommendersystem.recsys import recsys
from utils.LogPrint import Logger
from utils.build_train_rec_data_loader_new import *
from utils.build_test_rec_data_loader_new import *
from utils.global_variable import *
from utils import user_att_count
from utils import user_att_count_yelp
from utils import utils
from utils.config_yelp import ConfigYelp
from utils.config_last_fm import ConfigLastFM
from utils.config_last_fm_star import ConfigLastFMStar
from utils.config_yelp_star import ConfigYelpStar


use_gpu = True
config = None
rec = None

train_info_path = None

item_optimizer = None
att_att_optimizer = None
item_att_optimizer = None
test_info = None
add_neg_item_num_max = 100


def train_one_epoch(epoch_num):
    rec.init_train()
    train_loader = build_item_att_loader(config.name, train_info_path, config.user_info, config.item_info, config.att_tree_dict, \
                                         config.att_num, config.item_num, epoch_num,
                                         use_gpu=use_gpu, batch_size=config.batch_size, shuffle=config.train_shuffle,
                                         num_threads=config.num_threads)
    epoch_item_loss_sum = 0.
    epoch_att_att_loss_sum = 0.
    epoch_item_att_loss_sum = 0.
    epoch_count = 0
    for batch_data in tqdm(train_loader, ncols=0):
        user_list, pos_item_list, \
        item_pos_att_list, item_pos_att_mask, item_neg_att_list, item_neg_att_mask, \
        neg_item_list1, neg_item_mask1, neg_item_list2, neg_item_mask2, \
        att_pos_att_list, att_pos_att_mask, att_neg_att_list, att_neg_att_mask, att_pos_train_list, att_neg_train_list, \
        att_list, pos_list, neg_list, item_neg_item_list, item_neg_item_list_mask = batch_data

        if use_gpu:
            user_list = user_list.cuda()
            pos_item_list = pos_item_list.cuda()
            item_pos_att_list = item_pos_att_list.cuda()
            item_pos_att_mask = item_pos_att_mask.cuda()
            item_neg_att_list = item_neg_att_list.cuda()
            item_neg_att_mask = item_neg_att_mask.cuda()
            neg_item_list1 = neg_item_list1.cuda()
            neg_item_mask1 = neg_item_mask1.cuda()
            neg_item_list2 = neg_item_list2.cuda()
            neg_item_mask2 = neg_item_mask2.cuda()

            att_pos_att_list = att_pos_att_list.cuda()
            att_pos_att_mask = att_pos_att_mask.cuda()
            att_neg_att_list = att_neg_att_list.cuda()
            att_neg_att_mask = att_neg_att_mask.cuda()
            att_pos_train_list = att_pos_train_list.cuda()
            att_neg_train_list = att_neg_train_list.cuda()

            att_list = att_list.cuda()
            pos_list = pos_list.cuda()
            neg_list = neg_list.cuda()

            item_neg_item_list = item_neg_item_list.cuda()
            item_neg_item_list_mask = item_neg_item_list_mask.cuda()

        item_att_loss = 0.5 * rec.att_item_one_step_train(config.adj_index, att_list, pos_list, neg_list)
        item_att_loss_float = item_att_loss.cpu().detach().item()
        epoch_item_att_loss_sum += item_att_loss_float

        item_loss = rec.item_one_step_train(user_list, config.adj_index,
                                            item_pos_att_list, item_pos_att_mask,
                                            item_neg_att_list, item_neg_att_mask,
                                            pos_item_list, neg_item_list1, neg_item_mask1,
                                            neg_item_list2, neg_item_mask2, item_neg_item_list, item_neg_item_list_mask)
        item_loss_float = item_loss.cpu().detach().item()
        epoch_item_loss_sum += item_loss_float

        att_att_loss = rec.att_one_step_train(user_list, config.adj_index,
                                            att_pos_att_list, att_pos_att_mask,
                                              att_neg_att_list, att_neg_att_mask,
                                              att_pos_train_list, att_neg_train_list)
        att_att_loss_float = att_att_loss.cpu().detach().item()
        epoch_att_att_loss_sum += att_att_loss_float

        all_loss = item_loss + att_att_loss
        item_att_optimizer.zero_grad()
        all_loss.backward()
        item_att_optimizer.step()

        epoch_count += 1
        print("{} step item_loss: {} att_att loss: {} item_att_loss: {}".format(str(epoch_count), str(item_loss_float),
              str(att_att_loss_float), str(item_att_loss_float)))

    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch_item_loss = epoch_item_loss_sum / epoch_count
    epoch_att_att_loss = epoch_att_att_loss_sum / epoch_count
    epoch_item_att_loss = epoch_item_att_loss_sum / epoch_count
    print("{} train att epoch {} att_att loss: {} item_loss {} item_att_loss {}".format(time_str, str(epoch_num), str(epoch_item_loss),
           str(epoch_att_att_loss), str(epoch_item_att_loss)))
    return epoch_item_loss, epoch_att_att_loss, epoch_item_att_loss


def rec_test(test_model_path=None):
    if test_model_path is not None:
        if use_gpu:
            rec.rec.load_state_dict(torch.load(test_model_path))
        else:
            rec.rec.load_state_dict(torch.load(test_model_path, map_location='cpu'))
    rec.init_eval(config.adj_index)

    test_batch_size = 512
    test_shuffle = False
    test_num_threads = 2

    item_auc_list = []
    att_auc_list = []
    global_item_auc_list = []
    item_att_auc_list = []

    test_loader = build_test_item_att_loader(config.name, test_info, config.user_info, config.item_info, config.att_tree_dict,
                                             config.att_num, config.item_num, use_gpu=use_gpu,
                                             batch_size=test_batch_size, shuffle=test_shuffle,
                                             num_threads=test_num_threads)
    for batch_data in tqdm(test_loader, ncols=0):
        user_list, item_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, \
        test_item_list, test_item_mask, test_item_len, test_item_label, \
        test_attribute_list, test_attribute_mask, test_attribute_len, test_attribute_label, \
        global_test_item_list, global_test_item_mask, global_test_item_len, global_test_item_label, \
        all_att, all_att_label = batch_data

        if use_gpu:
            user_list = user_list.cuda()
            item_list = item_list.cuda()
            pos_att_list = pos_att_list.cuda()
            pos_att_mask = pos_att_mask.cuda()
            neg_att_list = neg_att_list.cuda()
            neg_att_mask = neg_att_mask.cuda()
            test_item_list = test_item_list.cuda()
            test_item_mask = test_item_mask.cuda()
            test_attribute_list = test_attribute_list.cuda()
            test_attribute_mask = test_attribute_mask.cuda()
            global_test_item_list = global_test_item_list.cuda()
            global_test_item_mask = global_test_item_mask.cuda()
            all_att = all_att.cuda()
            all_att_label = all_att_label.cuda()

        item_score = rec.rec.get_item_list_score(user_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask,
                                                             test_item_list)
        item_score = item_score.masked_fill(test_item_mask == False, 0.)
        attribute_score = rec.rec.get_att_list_score(user_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, test_attribute_list)
        attribute_score_list = attribute_score.masked_fill(test_attribute_mask == False, -1e9)
        global_item_score = rec.rec.get_item_list_score(user_list, pos_att_list, pos_att_mask, neg_att_list,
                                                                    neg_att_mask, global_test_item_list)
        global_item_score = global_item_score.masked_fill(global_test_item_mask == False, 0.)
        item_all_att_score = rec.rec.get_item_all_att_score(item_list, all_att)

        item_score_list = item_score.cpu().detach().numpy()
        attribute_score_list = attribute_score_list.cpu().detach().numpy()
        global_item_score_list = global_item_score.cpu().detach().numpy()
        item_all_att_score = item_all_att_score.cpu().detach().numpy()
        all_att_label = all_att_label.cpu().detach().numpy()

        for each_item_score, item_len, item_label in zip(item_score_list, test_item_len, test_item_label):
            each_item_score_ = each_item_score[:item_len]
            if len(set(item_label)) == 2:
                auc = roc_auc_score(item_label, each_item_score_)
                item_auc_list.append(auc)

        for each_att_score, att_len, att_label in zip(attribute_score_list, test_attribute_len, test_attribute_label):
            if sum(att_label) == 0:
                continue
            each_att_score_ = each_att_score[:att_len]
            auc = roc_auc_score(att_label, each_att_score_)
            att_auc_list.append(auc)

        for each_item_score, item_len, item_label in zip(global_item_score_list, global_test_item_len,
                                                         global_test_item_label):
            each_item_score_ = each_item_score[:item_len]
            if len(set(item_label)) == 2:
                auc = roc_auc_score(item_label, each_item_score_)
                global_item_auc_list.append(auc)

        for each_item_score, item_label in zip(item_all_att_score, all_att_label):
            auc = roc_auc_score(item_label, each_item_score)
            item_att_auc_list.append(auc)

    mean_item_auc = np.mean(np.array(item_auc_list))
    mean_global_item_auc = np.mean(np.array(global_item_auc_list))
    mean_item_att_auc = np.mean(np.array(item_att_auc_list))

    print("---------test-----------")
    print("item_auc: {}, global item_auc: {}, item_att_auc: {}".format(str(mean_item_auc),
                                                                        str(mean_global_item_auc),
                                                                       str(mean_item_att_auc)))
    return mean_item_auc, mean_global_item_auc, mean_item_att_auc


def main():
    global config
    global rec
    global train_info_path
    global item_optimizer
    global att_att_optimizer
    global item_att_optimizer
    global test_info

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {lastfm, lastfm_star, yelp, yelp_star}.')
    args = parser.parse_args()

    if args.data_name == LAST_FM:
        config = ConfigLastFM()
        user_att_count.set_config(config)
    elif args.data_name == YELP:
        config = ConfigYelp()
        user_att_count_yelp.set_config(config)
    elif args.data_name == LAST_FM_STAR:
        config = ConfigLastFMStar()
        user_att_count.set_config(config)
    elif args.data_name == YELP_STAR:
        config = ConfigYelpStar()
        user_att_count.set_config(config)

    rec = recsys(config, use_gpu=use_gpu)

    data_name = args.data_name
    if data_name == YELP_STAR:
        data_name = YELP

    train_info_path = "./data/" + data_name + '/' + FM_SAMPLE_DATA + '/sample_fm_data_train-'

    test_info_path = "./data/" + data_name + '/' + FM_SAMPLE_DATA + "/sample_fm_data_valid.pkl"
    test_info = utils.load_pickle(test_info_path)

    date_str = datetime.date.today().isoformat()
    sys.stdout = Logger("offline-train-rec-{}-ilr-{}-alr-{}-reg-{}-bs-{}-{}.log" \
                        .format(date_str, str(config.item_lr), str(config.att_lr), str(config.weight_decay),
                                str(config.batch_size),
                                rec.model_info_str))

    item_optimizer = optim.Adam([param for param in rec.rec.parameters() if param.requires_grad == True], \
                                lr=config.item_lr, weight_decay=config.weight_decay)
    att_att_optimizer = optim.Adam([param for param in rec.rec.parameters() if param.requires_grad == True], \
                               lr=config.att_lr, weight_decay=config.weight_decay)
    item_att_optimizer = optim.Adam([param for param in rec.rec.parameters() if param.requires_grad == True], \
                                    lr=config.att_lr, weight_decay=config.weight_decay)

    for _ in range(config.epoch_num):
        train_one_epoch(_)
        rec.save_model("iter{}-{}".format(str(_), date_str))
        if (_ + 1) % config.test_epoch_num == 0:
            with torch.no_grad():
                rec_test()

if __name__ == '__main__':
    main()
