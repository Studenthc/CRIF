import json
import pickle
import os
from utils.global_variable import *
from utils.config_yelp import ConfigYelp

config = None
global_att_length_att_not_subset_dict = None
att_single_group_item_id_not_subset_dict = None


def set_config(global_config):
    global config
    config = global_config
    _init()


def _init():
    global global_att_length_att_not_subset_dict
    global att_single_group_item_id_not_subset_dict

    f = open('utils/' + config.name + '/att_not_subset_item_id.txt', 'rb')
    att_single_group_item_id_not_subset_dict = pickle.load(f)
    f.close()

    f1 = open('utils/' + config.name + '/att_group_len_dict.txt', 'rb')
    global_att_length_att_not_subset_dict = pickle.load(f1)
    f1.close()


def get_item_att(item):
    return config.item_info[item]


def get_item_large_att(item):
    return config.large_item_info[item]


def get_item_att_tuple(item):
    return tuple(config.item_info[item])


def get_item_large_att_tuple(item):
    return tuple(config.large_item_info[item])


def get_att_item(att):
    return config.att_info[att]


def get_large_att_item(att):
    return config.large_att_info[att]


def get_item_id_for_single_att_group(att_group):
    return att_single_group_item_id_not_subset_dict[att_group]


def att_single_available_candidate_for_group(pos_att, neg_att):
    target_att_length = len(pos_att) + 1
    if global_att_length_att_not_subset_dict.get(target_att_length) is None:
        return dict()
    att_group_candidate = global_att_length_att_not_subset_dict[target_att_length]
    att_item_candidate = dict()
    for att_group in att_group_candidate:
        if set(pos_att).issubset(set(att_group)) and len(set(att_group) & set(neg_att)) == 0:
            item_id = get_item_id_for_single_att_group(att_group)
            att_item_candidate[item_id] = (set(att_group) - set(pos_att)).pop()
    return att_item_candidate