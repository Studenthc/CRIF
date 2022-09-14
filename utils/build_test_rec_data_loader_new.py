import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

import utils.user_att_count as user_att_count
import utils.user_att_count_yelp as user_att_count_yelp
from utils.global_variable import *

random.seed(1640)


def pad_list_of_list(list_of_list, pad_idx=0):
    maxlen = max([len(_) for _ in list_of_list])
    if maxlen == 0:
        maxlen = 1
    padded_list_of_list = np.full((len(list_of_list), maxlen), pad_idx)
    mask_list_of_list = np.full((len(list_of_list), maxlen), False)
    for i in range(len(list_of_list)):
        padded_list_of_list[i][:len(list_of_list[i])] = list_of_list[i]
        mask_list_of_list[i][:len(list_of_list[i])] = True
    return torch.from_numpy(padded_list_of_list), torch.from_numpy(mask_list_of_list)


def test_item_att_collate_fn(batch):
    user, item, pos_att, neg_att, \
    test_item, test_item_len, test_item_label, \
    test_attribute, test_attribute_len, test_attribute_label, \
    global_test_item, global_test_item_len, global_test_item_label, \
    item_all_att, item_all_att_label = zip(*batch)

    user_list = torch.tensor(user)
    item_list = torch.tensor(item)
    pos_att_list, pos_att_mask = pad_list_of_list(pos_att)
    neg_att_list, neg_att_mask = pad_list_of_list(neg_att)
    test_attribute_list, test_attribute_mask = pad_list_of_list(test_attribute)
    test_item_list, test_item_mask = pad_list_of_list(test_item)
    global_test_item_list, global_test_item_mask = pad_list_of_list(global_test_item)
    item_all_att_list, item_all_att_list_mask = pad_list_of_list(item_all_att)
    item_all_att_label_list, item_all_att_label_mask = pad_list_of_list(item_all_att_label)

    return user_list, item_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, \
           test_item_list, test_item_mask, test_item_len, test_item_label, \
           test_attribute_list, test_attribute_mask, test_attribute_len, test_attribute_label, \
           global_test_item_list, global_test_item_mask, global_test_item_len, global_test_item_label, \
           item_all_att_list, item_all_att_label_list


class ItemAttTestGenerator(Dataset):
    def __init__(self, name, test_info, user_info, item_info, att_tree_dict,
                 att_num, item_num, use_gpu=False):
        self.data_name = name
        self.test_info = test_info
        self.user_info = user_info
        for key in self.user_info:
            self.user_info[key] = set(self.user_info[key])
        self.item_info = item_info
        for key in self.item_info:
            self.item_info[key] = set(self.item_info[key])
        self.att_tree_dict = att_tree_dict
        for key in self.att_tree_dict:
            self.att_tree_dict[key] = set(self.att_tree_dict[key])
        self.att_num = att_num
        self.item_num = item_num
        self.len = len(self.test_info[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        user = self.test_info[0][index]
        item = self.test_info[1][index]
        pos_att_len = len(self.item_info[item])
        pos_number = 1
        if pos_att_len > 1:
            pos_number = random.sample(range(1, pos_att_len), 1)[0]
        if pos_number > 5:
            pos_number = 5
        neg_number = random.sample(range(1, 4), 1)[0]
        item_pos_att = random.sample(self.item_info[item], pos_number)
        item_neg_att = random.sample(set(range(self.att_num)) - self.item_info[item], neg_number)
        asked_list = item_pos_att + item_neg_att
        pos_att = item_pos_att
        neg_att = item_neg_att

        pos_list = []
        neg_list = []
        easy_neg_list = []
        candidate_list = []
        for k in range(self.item_num):
            k_att = None
            if self.data_name == YELP:
                k_att = user_att_count_yelp.get_item_att(k)
            else:
                k_att = user_att_count.get_item_att(k)
            if set(item_pos_att).issubset(k_att) and (len(set(item_neg_att) & k_att) == 0):
                pos_list.append(k)
                candidate_list.append(k)
            elif set(item_pos_att).issubset(k_att):
                neg_list.append(k)
                candidate_list.append(k)
            else:
                easy_neg_list.append(k)

        # -------------item--------------
        candidate_list.remove(item)
        test_item = [item] + candidate_list[:300]
        test_item_len = len(test_item)
        test_item_label = [1] + [0] * len(candidate_list[:300])

        # -------------att att--------------
        full_attribute = self.item_info[item]
        test_pos_attribute = list(full_attribute - set(pos_att))
        test_neg_attribute = list(set(range(self.att_num)) - full_attribute)
        test_attribute = test_pos_attribute + test_neg_attribute
        test_attribute_len = len(test_attribute)
        test_attribute_label = [1] * len(test_pos_attribute) + [0] * len(test_neg_attribute)

        # -------------BPR item------------
        if len(pos_list) >= 100:
            pos_list = random.sample(pos_list, 100)
        if len(neg_list) >= 100:
            neg_list = random.sample(neg_list, 100)
        global_test_item = pos_list + neg_list
        global_test_item_len = len(global_test_item)
        global_test_item_label = [1] * len(pos_list) + [0] * len(neg_list)

        # -------------BPR item att------------
        pos_att_item_att = self.item_info[item]
        item_all_att = range(self.att_num)
        item_all_att_label = []
        for i in range(self.att_num):
            if i in pos_att_item_att:
                item_all_att_label.append(1)
            else:
                item_all_att_label.append(0)

        return user, item, pos_att, neg_att, \
               test_item, test_item_len, test_item_label, \
               test_attribute, test_attribute_len, test_attribute_label, \
               global_test_item, global_test_item_len, global_test_item_label, \
               item_all_att, item_all_att_label


def build_test_item_att_loader(name, test_info, user_info, item_info, att_tree_dict,
                               att_num, item_num, use_gpu=True,
                               batch_size=1, shuffle=True, num_threads=4):
    test_generator = ItemAttTestGenerator(name, test_info, user_info, item_info, att_tree_dict,
                                          att_num, item_num, use_gpu=use_gpu)
    return DataLoader(
        test_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=test_item_att_collate_fn
    )
