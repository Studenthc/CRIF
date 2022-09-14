import os

import torch

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from utils.global_variable import *
import json
from utils.utils import *


class ConfigYelpStar:
    def __init__(self):
        self.name = 'yelp_star'
        self.use_gpu = True
        self.user_num = 27675
        self.item_num = 70311
        self.attribute_num = 590
        self.parent_attribute_num = 590
        self.nlayer = 2
        self.conv_name = 'gcn'
        self.n_heads = 1
        self.drop = 0.1
        self.max_rec_item_num = 10
        self.top_taxo = 3
        self.feedback_aggregate = "3loss_not_equal"
        self.layer_aggregate = "mean"

        self.att_num = self.attribute_num

        self.batch_size = 1024
        self.item_lr = 0.001
        self.att_lr = 0.0003
        self.weight_decay = 1e-5
        self.train_shuffle = True
        self.num_threads = 4
        self.add_neg_item_num_max = 100
        self.epoch_num = 10000
        self.test_epoch_num = 10000

        self.input_dim = 4 + 15 + 8
        self.hidden_dim = 64
        self.output_dim = 2

        self.rec_model_path = root_path + "/recommendersystem/recmodel/yelp"

        self.user_info = None
        self.item_info = None
        self.att_info = None
        self.att_reverse_info = None
        self.adj_index = [[], []]
        self.att_tree_dict = None
        self._data_init()

    def _data_init(self):
        with open("./data/" + 'yelp' + '/' + GRAPH_GENERATE_DATA + '/' + "/user_item.json", "r") as f:
            user_info = json.load(f)
        new_user_info = {}
        for user in user_info:
            new_user_info[int(user)] = set(user_info[user])
        self.user_info = new_user_info

        with open("./data/" + 'yelp' + '/' + GRAPH_GENERATE_DATA + '/' + "/item_dict-original_tag.json", "r") as f:
            item_info = json.load(f)
        new_item_info = {}
        att_info = {}
        att_reverse_info = {}

        need_att_info = False
        if os.path.exists('utils/' + 'yelp' + '/att_reverse_info.txt'):
            f = open('utils/' + 'yelp' + '/att_info.txt', 'rb')
            att_info = pickle.load(f)
            f.close()

            f1 = open('utils/' + 'yelp' + '/att_reverse_info.txt', 'rb')
            att_reverse_info = pickle.load(f1)
            f1.close()
        else:
            need_att_info = True

        for item in item_info:
            new_item_info[int(item)] = set(item_info[item][ATT_KEY])

            if need_att_info:
                for att in item_info[item][ATT_KEY]:
                    if len(att_info.get(int(att), set())) == 0:
                        att_info[int(att)] = set()
                        att_info[int(att)].add(int(item))
                    else:
                        att_info[int(att)].add(int(item))

                for att in set(range(self.att_num)) - set(item_info[item][ATT_KEY]):
                    if len(att_reverse_info.get(int(att), set())) == 0:
                        att_reverse_info[int(att)] = list()
                        att_reverse_info[int(att)].append(int(item))
                    else:
                        att_reverse_info[int(att)].append(int(item))

        if need_att_info:
            att_info_file = open('utils/' + 'yelp' + '/att_info.txt', 'wb')
            pickle.dump(att_info, att_info_file)
            att_info_file.close()
            att_reverse_info_file = open('utils/' + 'yelp' + '/att_reverse_info.txt', 'wb')
            pickle.dump(att_reverse_info, att_reverse_info_file)
            att_reverse_info_file.close()

        self.item_info = new_item_info
        self.att_info = att_info
        self.att_reverse_info = att_reverse_info

        for user in self.user_info:
            for item in self.user_info[user]:
                self.adj_index[0].append(user)
                self.adj_index[1].append(item + self.user_num)
                self.adj_index[1].append(user)
                self.adj_index[0].append(item + self.user_num)
        for item in self.item_info:
            for att in self.item_info[item]:
                self.adj_index[0].append(item + self.user_num)
                self.adj_index[1].append(att + self.item_num + self.user_num)
                self.adj_index[1].append(item + self.user_num)
                self.adj_index[0].append(att + self.item_num + self.user_num)
        self.adj_index = torch.tensor(self.adj_index)

        if self.use_gpu:
            self.adj_index = self.adj_index.cuda()

        with open("./data/" + 'yelp' + '/' + GRAPH_GENERATE_DATA + '/' + "/yelp_star_attribute_tree_dict.json", "r") as f:
            attribute_tree_dict = json.load(f)
        self.att_tree_dict = trans_index(attribute_tree_dict)
