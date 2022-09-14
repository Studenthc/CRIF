import json
import pickle
import os
from utils.global_variable import *
from utils.config_yelp import ConfigYelp

config = None
global_att_length_att_dict = None
att_single_group_item_id_dict = None


def set_config(global_config):
    global config
    config = global_config
    _init()


def _init():
    global global_att_length_att_dict
    global att_single_group_item_id_dict

    if os.path.exists('utils/' + config.name + '/att_single_group_item_id.txt'):
        f = open('utils/' + config.name + '/att_single_group_item_id.txt', 'rb')
        att_single_group_item_id_dict = pickle.load(f)
        f.close()

        f1 = open('utils/' + config.name + '/att_single_group_length_dict.txt', 'rb')
        global_att_length_att_dict = pickle.load(f1)
        f1.close()
    else:
        global_att_length_count_dict, global_att_length_att_dict = att_single_group_length_dict()
        att_single_group_item_id_dict = att_single_group_item_id_table()

        att_not_subset_item_id_file = open('utils/' + config.name + '/att_single_group_item_id.txt', 'wb')
        pickle.dump(att_single_group_item_id_dict, att_not_subset_item_id_file)
        att_not_subset_item_id_file.close()

        att_group_len_file = open('utils/' + config.name + '/att_single_group_length_dict.txt', 'wb')
        pickle.dump(global_att_length_att_dict, att_group_len_file)
        att_group_len_file.close()


def get_item_att(item):
    return config.item_info[item]


def get_item_att_tuple(item):
    return tuple(config.item_info[item])


def get_att_item(att):
    return config.att_info[att]


def get_att_item_list(att):
    return list(config.att_info[att])


def get_att_reverse_item(att):
    return config.att_reverse_info[att]


def att_group_count():
    att_group_dict = dict()
    for item_id in range(config.item_num):
        att = get_item_att_tuple(item_id)
        count = att_group_dict.get(att, 0) + 1
        att_group_dict[att] = count
    return att_group_dict

def att_single_group_length_dict():
    att_single_group_length_count_dict = dict()
    att_single_group_length_att_dict = dict()
    att_group_dict = att_group_count()
    for item_id in range(config.item_num):
        att = get_item_att_tuple(item_id)
        if att_group_dict[att] >= 1:
            single_group_length = len(att)
            count = att_single_group_length_count_dict.get(single_group_length, 0) + 1
            att_single_group_length_count_dict[single_group_length] = count
            if att_single_group_length_att_dict.get(single_group_length) is None:
                att_single_group_length_att_dict[single_group_length] = list()
            att_single_group_length_att_dict[single_group_length].append(att)
    return att_single_group_length_count_dict, att_single_group_length_att_dict


def att_single_group_item_id_table():
    att_single_group_item_id = dict()
    att_group_dict = att_group_count()
    for item_id in range(config.item_num):
        att = get_item_att_tuple(item_id)
        if att_group_dict[att] >= 1:
            att_single_group_item_id[att] = item_id
    return att_single_group_item_id


def get_item_id_for_single_att_group(att_group):
    return att_single_group_item_id_dict[att_group]


def att_single_available_candidate_for_group(pos_att, neg_att):
    target_att_length = len(pos_att) + 1
    if global_att_length_att_dict.get(target_att_length) is None:
        return dict()
    att_group_candidate = global_att_length_att_dict[target_att_length]
    att_item_candidate = dict()
    for att_group in att_group_candidate:
        if set(pos_att).issubset(set(att_group)) and len(set(att_group) & set(neg_att)) == 0:
            item_id = get_item_id_for_single_att_group(att_group)
            att_item_candidate[item_id] = (set(att_group) - set(pos_att)).pop()
    return att_item_candidate
