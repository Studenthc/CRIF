import json
import random

import numpy as np
import torch
import torch.nn as nn
from recommendersystem.myrecmodel import MyRec


def pad_list_of_list(list_of_list, pad_idx=0):
    maxlen = max([len(_) for _ in list_of_list])
    padded_list_of_list = np.full((len(list_of_list), maxlen), pad_idx)
    mask_list_of_list = np.full((len(list_of_list), maxlen), False)
    for i in range(len(list_of_list)):
        padded_list_of_list[i][:len(list_of_list[i])] = list_of_list[i]
        mask_list_of_list[i][:len(list_of_list[i])] = True
    return torch.from_numpy(padded_list_of_list), torch.from_numpy(mask_list_of_list)


class recsys():
    def __init__(self, config, convhis=None, use_gpu=None):
        if use_gpu is not None:
            config.use_gpu = use_gpu

        self.model_info_str = 'dim-{}_layer-{}_conv-{}_fa-{}_la-{}'.\
                            format(str(config.hidden_dim), str(config.nlayer), str(config.conv_name), \
                                    str(config.feedback_aggregate), str(config.layer_aggregate))

        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.use_gpu = config.use_gpu

        self.item_offset = torch.tensor(0)
        self.att_offset = torch.tensor(self.item_num)

        self.max_rec_item_num = config.max_rec_item_num
        self.convhis = convhis

        self.attribute_tree = config.att_tree_dict
        new_attribute_tree = {}
        for parent in self.attribute_tree:
            new_attribute_tree[int(parent)] = set(self.attribute_tree[parent])
        self.attribute_tree = new_attribute_tree

        self.attribute_parent_mat = np.zeros([self.parent_attribute_num, self.attribute_num])
        for attribute_parent, attribute_list in self.attribute_tree.items():
            for attribute in attribute_list:
                self.attribute_parent_mat[attribute_parent, attribute] = 1.

        self.logsigmoid = nn.LogSigmoid()
        self.tanh = nn.Tanh()

        self.rec = MyRec(config)
        self.rec_model_path = config.rec_model_path

        if self.use_gpu:
            self.item_offset = self.item_offset.cuda()
            self.att_offset = self.att_offset.cuda()

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.mse_criterion = torch.nn.MSELoss(reduce=True, size_average=True)
        self.triplet_loss = nn.TripletMarginLoss(margin=2.0, p=2)
        self.ranking_loss = nn.MarginRankingLoss(margin=2.0)

    def save_model(self, extra_name=None):
        name = "rec_model" + "_" + self.model_info_str
        if extra_name is not None:
            name = name + '_' + extra_name
        torch.save(self.rec.state_dict(), "/".join([self.rec_model_path, name]))

    def load_model(self, extra_name=None, transfer_to_cpu=False):
        name = "rec_model" + "_" + self.model_info_str
        if extra_name is not None:
            name = name + '_' + extra_name
        if transfer_to_cpu:
            self.rec.load_state_dict(torch.load("/".join([self.rec_model_path, name]), map_location='cpu'))
        else:
            self.rec.load_state_dict(torch.load("/".join([self.rec_model_path, name])))

    def init_train(self):
        self.rec.train()

    def init_eval(self, edge_index):
        self.rec.eval()
        self.rec.graph_prop(edge_index)

    def get_item_preference(self, user, pos_attribute, neg_attribute, candidate_list=None):
        pos_att_list = []
        neg_att_list = []

        for _ in pos_attribute:
            pos_att_list.append(_)

        for _ in neg_attribute:
            neg_att_list.append(_)

        user = torch.tensor(user)
        pos_att_list = torch.tensor(pos_att_list)
        neg_att_list = torch.tensor(neg_att_list)
        if candidate_list is not None:
            candidate_list = torch.tensor(candidate_list)

        if self.use_gpu:
            user = user.cuda()
            pos_att_list = pos_att_list.cuda()
            neg_att_list = neg_att_list.cuda()
            if candidate_list is not None:
                candidate_list = candidate_list.cuda()

        neg_item_list = []
        target_item = -1
        item_score = self.rec.get_item_score_single(user, target_item, pos_att_list, neg_att_list, neg_item_list, candidate_list)
        item_score = item_score.detach()
        if self.use_gpu:
            item_score = item_score.cpu()
        return item_score

    def get_item_preference_with_neg_item(self, user, target_item, pos_attribute, neg_attribute, neg_item_list, candidate_list=None):
        pos_att_list = []
        neg_att_list = []

        for _ in pos_attribute:
            pos_att_list.append(_)

        for _ in neg_attribute:
            neg_att_list.append(_)

        user = torch.tensor(user)
        pos_att_list = torch.tensor(pos_att_list)
        neg_att_list = torch.tensor(neg_att_list)
        if candidate_list is not None:
            candidate_list = torch.tensor(candidate_list)

        if self.use_gpu:
            user = user.cuda()
            pos_att_list = pos_att_list.cuda()
            neg_att_list = neg_att_list.cuda()
            if candidate_list is not None:
                candidate_list = candidate_list.cuda()

        item_score = self.rec.get_item_score_single(user, target_item, pos_att_list, neg_att_list, neg_item_list, candidate_list)
        item_score = item_score.detach()
        if self.use_gpu:
            item_score = item_score.cpu()
        return item_score

    def item_one_step_train(self, user_list, edge_index, known_pos_att_list, known_pos_att_mask,
                            known_neg_att_list, known_neg_att_mask, pos_item_list, neg_item_list1,
                            neg_item_mask1, neg_item_list2, neg_item_mask2, known_neg_item_list, known_neg_item_mask):

        self.rec.graph_prop(edge_index)

        self.rec.update_neg_user_embedding(user_list, pos_item_list, known_neg_item_list, known_neg_item_mask)

        pos_item_score = self.rec.get_item_score(user_list, known_pos_att_list, known_pos_att_mask,
                                                 known_neg_att_list, known_neg_att_mask, pos_item_list.unsqueeze(-1)).squeeze(-1)
        neg_item_score1 = self.rec.get_item_score(user_list, known_pos_att_list, known_pos_att_mask,
                                                 known_neg_att_list, known_neg_att_mask, neg_item_list1)
        neg_item_score1 = neg_item_score1.masked_fill(neg_item_mask1 == False, 0.)
        neg_item_score1, _ = neg_item_score1.max(dim=-1)
        neg_item_score2 = self.rec.get_item_score(user_list, known_pos_att_list, known_pos_att_mask,
                                                 known_neg_att_list, known_neg_att_mask, neg_item_list2)
        neg_item_score2 = neg_item_score2.masked_fill(neg_item_mask2 == False, 0.)
        neg_item_score2, _ = neg_item_score2.max(dim=-1)

        item_loss1 = - self.logsigmoid(pos_item_score - neg_item_score1)
        item_loss2 = - self.logsigmoid(pos_item_score - neg_item_score2)
        item_loss1 = item_loss1.mean()
        item_loss2 = item_loss2.mean()
        item_loss = item_loss1 + item_loss2
        return item_loss

    def att_one_step_train(self, user_list, edge_index, known_pos_att_list, known_pos_att_mask,
                            known_neg_att_list, known_neg_att_mask,
                            att_pos_train_list, att_neg_train_list):
        self.rec.graph_prop(edge_index)

        pos_att_score = self.rec.get_att_score(user_list, known_pos_att_list, known_pos_att_mask,
                                                 known_neg_att_list, known_neg_att_mask, att_pos_train_list.unsqueeze(-1)).squeeze(-1)
        neg_att_score = self.rec.get_att_score(user_list, known_pos_att_list, known_pos_att_mask,
                                                 known_neg_att_list, known_neg_att_mask, att_neg_train_list.unsqueeze(-1)).squeeze(-1)

        att_loss = - (self.logsigmoid(pos_att_score - neg_att_score)).mean()
        return att_loss

    def att_item_one_step_train(self, edge_index, att_list, pos_list, neg_list):

        self.rec.graph_prop(edge_index)

        pos_score = self.rec.get_att_item_score(att_list, pos_list)
        neg_score = self.rec.get_att_item_score(att_list, neg_list)

        item_att_loss = self.bce_criterion(pos_score, torch.ones_like(pos_score).cuda())
        item_att_loss += self.bce_criterion(neg_score, torch.zeros_like(neg_score).cuda())
        return item_att_loss

    def get_recommend_item_list(self, target_item, neg_item_list, candidate_list=None):
        user = self.convhis.get_user()
        pos_attribute = self.convhis.get_pos_attribute()
        neg_attribute = self.convhis.get_neg_attribute()

        if neg_item_list == None:
            item_score_list = self.get_item_preference(user, pos_attribute, neg_attribute, candidate_list)
        else:
            item_score_list = self.get_item_preference_with_neg_item(user, target_item, pos_attribute, neg_attribute, neg_item_list, candidate_list)
        values, indices = item_score_list.sort(descending=True)
        if candidate_list is None:
            return indices.tolist()[:self.max_rec_item_num]
        else:
            indices = indices.tolist()[:self.max_rec_item_num]
            item_list = []
            for i in indices:
                item_list.append(candidate_list[i])
            return item_list, item_score_list

    def get_recommend_number_item_list(self, candidate_list=None, number=0):
        user = self.convhis.get_user()
        pos_attribute = self.convhis.get_pos_attribute()
        neg_attribute = self.convhis.get_neg_attribute()

        item_score_list = self.get_item_preference(user, pos_attribute, neg_attribute, candidate_list)
        values, indices = item_score_list.sort(descending=True)
        if candidate_list is None:
            return indices.tolist()[:self.max_rec_item_num - number]
        else:
            indices = indices.tolist()[:self.max_rec_item_num - number]
            item_list = []
            for i in indices:
                item_list.append(candidate_list[i])
            return item_list

