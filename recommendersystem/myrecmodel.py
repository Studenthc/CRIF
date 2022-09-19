import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from recommendersystem.conv import GeneralConv
from utils.user_att_count import get_item_att
from utils.build_train_rec_data_loader_new import pad_list_of_list

class MyRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_aggregate = config.layer_aggregate
        self.gpu = config.use_gpu
        self.hidden_dim = config.hidden_dim
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.n_layers = config.nlayer
        self.attribute_num = config.attribute_num
        self.user_embed = nn.Embedding(self.user_num, self.hidden_dim)
        self.item_embed = nn.Embedding(self.item_num, self.hidden_dim)
        self.attribute_embed = nn.Embedding(self.attribute_num, self.hidden_dim)
        self.init_para()

        self.user_index = torch.tensor([_ for _ in range(self.user_num)])
        self.item_index = torch.tensor([_ for _ in range(self.item_num)])
        self.attribute_index = torch.tensor([_ for _ in range(self.attribute_num)])     
        self.user_graph_index = torch.tensor([_ for _ in range(self.user_num)])   
        self.item_graph_index = torch.tensor([_ for _ in range(self.user_num, self.user_num+self.item_num)])
        self.attribute_graph_index = torch.tensor([_ for _ in range(self.user_num+self.item_num, self.user_num+self.item_num+self.attribute_num)])

        self.gcs = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)
        self.nlayer = config.nlayer
        self.conv_name = config.conv_name
        self.n_heads = config.n_heads
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.tanh = nn.Tanh()
        for _ in range(self.n_layers):
            self.gcs.append(GeneralConv(self.conv_name, self.hidden_dim, self.hidden_dim, self.n_heads))
        self.gu_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.graph_rep = None
        self.eps = torch.tensor(1e-9)
        self.drop = nn.Dropout(config.drop)

        if self.layer_aggregate == 'last_layer' or self.layer_aggregate == 'mean':
            self.graph_dim = self.hidden_dim
        elif self.layer_aggregate == 'concat':
            self.graph_dim = (self.n_layers + 1) * self.hidden_dim
        else:
            print("not support layer_aggregate type : {} !!!".format(self.layer_aggregate))

        self.graph_layer_rep = None
        self.current_user = None
        self.neg_user = None

        if self.gpu:
            self.user_embed = self.user_embed.cuda()
            self.item_embed = self.item_embed.cuda()
            self.attribute_embed = self.attribute_embed.cuda()
            self.gcs = self.gcs.cuda()
            self.item_index = self.item_index.cuda()
            self.attribute_index = self.attribute_index.cuda()
            self.item_graph_index = self.item_graph_index.cuda()
            self.attribute_graph_index = self.attribute_graph_index.cuda()
            self.eps = self.eps.cuda()
            self.drop = self.drop.cuda()

            self.user_index = self.user_index.cuda()
            self.item_index = self.item_index.cuda()
            self.attribute_index = self.attribute_index.cuda()
            self.user_graph_index = self.user_graph_index.cuda()
            self.item_graph_index = self.item_graph_index.cuda()
            self.attribute_graph_index = self.attribute_graph_index.cuda()
            self.gu_linear = self.gu_linear.cuda()

    def init_para(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.user_num + self.item_num + self.attribute_num, self.hidden_dim)
        )
        nn.init.xavier_uniform_(all_embed)
        self.user_embed.weight.data = all_embed[:self.user_num].data
        self.item_embed.weight.data = all_embed[self.user_num:self.user_num+self.item_num].data
        self.attribute_embed.weight.data = all_embed[self.user_num+self.item_num:self.user_num+self.item_num+self.attribute_num].data

    def graph_prop(self, edge_index):
        node_input_user = self.user_embed(self.user_index)
        node_input_item = self.item_embed(self.item_index)
        node_input_att = self.attribute_embed(self.attribute_index)
        node_input = torch.cat([node_input_user, node_input_item, node_input_att], dim=0)
        self.graph_layer_rep = [node_input]
        for gc in self.gcs:
            x = gc(self.graph_layer_rep[-1], edge_index.cuda())
            x = F.leaky_relu(x)
            x = self.drop(x)
            self.graph_layer_rep.append(x)

        if self.layer_aggregate == 'last_layer':
            self.graph_rep = self.graph_layer_rep[-1]
        if self.layer_aggregate == 'mean':
            graph_layer_rep_tensor = torch.stack(self.graph_layer_rep, dim=1)
            self.graph_rep = torch.mean(graph_layer_rep_tensor, dim=1)
        if self.layer_aggregate == 'concat':
            self.graph_rep = torch.cat(self.graph_layer_rep, dim=1)

        return self.graph_rep

    def get_user_embedding(self, user):
        e_u = self.graph_rep[self.user_graph_index[user]]
        # e_u = self.user_embed[user]
        self.current_user = e_u
        if self.gpu:
            self.current_user = self.current_user.cuda()
        return self.current_user

    def get_item_score(self, user, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask, item):
        user_feature = self.neg_user
        item_feature = self.graph_rep[self.item_graph_index[item]]
        user_feature = user_feature.unsqueeze(-2).expand(item_feature.size())
        user_item_score = torch.sum(user_feature * item_feature, dim=-1)
        user_item_score = self.tanh(user_item_score)
        att_item_score = self.get_pos_neg_att_list_item_score(pos_att_list, pos_att_list_mask,
                                                              neg_att_list, neg_att_list_mask, item)
        att_item_score = att_item_score.unsqueeze(-1)
        item_score = user_item_score + att_item_score
        return item_score

    def get_item_list_score(self, user, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask, item_list):
        user_feature = self.get_user_embedding(user)
        item_feature = self.graph_rep[self.item_graph_index[item_list]]
        user_feature = user_feature.unsqueeze(-2).expand(item_feature.size())
        user_item_score = torch.sum(user_feature * item_feature, dim=-1)
        user_item_score = self.tanh(user_item_score)
        att_item_score = self.get_pos_neg_att_list_item_list_score(pos_att_list, pos_att_list_mask,
                                                              neg_att_list, neg_att_list_mask, item_list)
        item_score = user_item_score + att_item_score
        return item_score

    def update_neg_user_embedding(self, user_list, pos_item_list, neg_item_list, neg_item_mask):
        if self.gpu:
            neg_item_list = neg_item_list.cuda()
            neg_item_mask = neg_item_mask.cuda()
        e_u = self.graph_rep[self.user_graph_index[user_list]]

        neg_item_rep = self.graph_rep[self.item_graph_index[neg_item_list]]
        neg_item_rep_mask = neg_item_mask.unsqueeze(-1).expand(neg_item_rep.size())
        neg_item_rep = neg_item_rep.masked_fill(neg_item_rep_mask==False, 0.)

        neg_item_att_list = []
        for i, items in enumerate(neg_item_att_list):
            item_att_list = []
            for item in items:
                item_att_list.apend(list(get_item_att(item.item()).intersection(get_item_att(pos_item_list[i].item()))))
        for i, atts_list in enumerate(neg_item_att_list):
            for j, atts in enumerate(atts_list):
                if len(atts) != 0:
                    att_rep_list = self.graph_rep[self.attribute_graph_index[atts]]
                    for k in range(att_rep_list.size(0)):
                        att_rep_list[k] = self.gu_linear(att_rep_list[k].clone())
                    item_rep = neg_item_rep[i][j].expand(att_rep_list.size())
                    att_rep_list = att_rep_list * item_rep.clone()
                    att_item_score_list = []
                    for att in atts:
                        att_item_score_list.append(self.get_att_item_score(att, neg_item_list[i][j]).item())
                    att_item_score_list = torch.Tensor(att_item_score_list)
                    if self.gpu:
                        att_item_score_list = att_item_score_list.cuda()
                    att_item_score_list = att_item_score_list.unsqueeze(-1)
                    att_item_score = att_item_score_list.expand(att_rep_list.size())
                    pos_part = torch.sum(att_rep_list * att_item_score, dim=0) / len(atts)
                    neg_item_rep[i][j] = neg_item_rep[i][j] - pos_part
        
        self.neg_user = e_u - torch.sum(neg_item_rep, dim=-2) / len(neg_item_list[0])

    def get_item_score_single(self, user, target_item, pos_att_list, neg_att_list, neg_item, item_list=None):
        if neg_item == None:
            user_feature = self.get_user_embedding(user)
        else:
            neg_item_list, neg_item_list_mask = pad_list_of_list([neg_item])
            self.update_neg_user_embedding([user], [target_item], neg_item_list, neg_item_list_mask)
            user_feature = self.neg_user.squeeze(0)
        if item_list is None or len(item_list) == 0:
            raise ValueError
            pass
        else:
            item_feature = self.graph_rep[self.item_graph_index[item_list]]
            user_feature = user_feature.unsqueeze(-2).expand(item_feature.size())
            user_item_score = torch.sum(user_feature * item_feature, dim=-1)
            user_item_score = self.tanh(user_item_score)
            att_item_score = self.get_pos_neg_att_list_item_score_single(pos_att_list, neg_att_list, item_list)
            item_score = user_item_score + att_item_score
        return item_score


    def get_att_score(self, user, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask,
                       att):
        user_feature = self.get_user_embedding(user)
        att_feature = self.graph_rep[self.attribute_graph_index[att]]
        user_feature = user_feature.unsqueeze(-2).expand(att_feature.size())
        user_att_score = torch.sum(user_feature * att_feature, dim=-1)
        user_att_score = self.tanh(user_att_score)
        att_att_score = self.get_pos_neg_att_list_att_score(pos_att_list, pos_att_list_mask,
                                                              neg_att_list, neg_att_list_mask, att)
        att_att_score = att_att_score.unsqueeze(-1)
        att_score = user_att_score + att_att_score
        return att_score

    def get_att_list_score(self, user, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask,
                       att_list):
        user_feature = self.get_user_embedding(user)
        att_feature = self.graph_rep[self.attribute_graph_index[att_list]]
        user_feature = user_feature.unsqueeze(-2).expand(att_feature.size())
        user_att_score = torch.sum(user_feature * att_feature, dim=-1)
        user_att_score = self.tanh(user_att_score)
        att_att_score = self.get_pos_neg_att_list_att_list_score(pos_att_list, pos_att_list_mask,
                                                              neg_att_list, neg_att_list_mask, att_list)
        att_score = user_att_score + att_att_score
        return att_score

    def get_att_item_score(self, att, item):
        att_rep = self.graph_rep[self.attribute_graph_index[att]]
        item_feature = self.graph_rep[self.item_graph_index[item]]
        att_item_score = torch.sum(att_rep * item_feature, dim=-1)
        att_item_score = self.tanh(att_item_score)
        return att_item_score

    def get_att_list_item_score(self, att_list, att_list_mask, item):
        att_rep = self.graph_rep[self.attribute_graph_index[att_list]]
        att_rep_mask = att_list_mask.unsqueeze(-1).expand(att_rep.size())
        att_rep = att_rep.masked_fill(att_rep_mask == False, 0.)

        item_feature = self.graph_rep[self.item_graph_index[item]]

        att_item_score = torch.sum(att_rep * item_feature, dim=-1)
        att_item_score = self.tanh(att_item_score)
        att_item_score = torch.sum(att_item_score, dim=-1)
        return att_item_score

    def get_att_list_item_list_score(self, att_list, att_list_mask, item_list):
        att_rep = self.graph_rep[self.attribute_graph_index[att_list]]
        att_rep_mask = att_list_mask.unsqueeze(-1).expand(att_rep.size())
        att_rep = att_rep.masked_fill(att_rep_mask == False, 0.)
        att_rep = att_rep.unsqueeze(-2)

        item_feature = self.graph_rep[self.item_graph_index[item_list]]
        item_feature = item_feature.unsqueeze(-3)

        att_item_score = torch.sum(att_rep * item_feature, dim=-1)
        att_item_score = self.tanh(att_item_score)
        att_item_score = torch.sum(att_item_score, dim=-2)
        return att_item_score

    def get_att_list_att_score(self, att_list, att_list_mask, score_att_list):
        att_rep = self.graph_rep[self.attribute_graph_index[att_list]]
        att_rep_mask = att_list_mask.unsqueeze(-1).expand(att_rep.size())
        att_rep = att_rep.masked_fill(att_rep_mask == False, 0.)

        score_att_feature = self.graph_rep[self.attribute_graph_index[score_att_list]]

        att_att_score = torch.sum(att_rep * score_att_feature, dim=-1)
        att_att_score = self.tanh(att_att_score)
        att_att_score = torch.sum(att_att_score, dim=-1)
        return att_att_score

    def get_att_list_att_list_score(self, att_list, att_list_mask, score_att_list):
        att_rep = self.graph_rep[self.attribute_graph_index[att_list]]
        att_rep_mask = att_list_mask.unsqueeze(-1).expand(att_rep.size())
        att_rep = att_rep.masked_fill(att_rep_mask == False, 0.)
        att_rep = att_rep.unsqueeze(-2)

        score_att_feature = self.graph_rep[self.attribute_graph_index[score_att_list]]
        score_att_feature = score_att_feature.unsqueeze(-3)

        att_att_score = torch.sum(att_rep * score_att_feature, dim=-1)
        att_att_score = self.tanh(att_att_score)
        att_att_score = torch.sum(att_att_score, dim=1)
        return att_att_score

    def get_att_list_item_score_single(self, att_list, item_list):
        att_rep = self.graph_rep[self.attribute_graph_index[att_list]]

        item_feature = self.graph_rep[self.item_graph_index[item_list]]

        att_rep = att_rep.unsqueeze(-2)
        item_feature = item_feature.unsqueeze(0)
        att_item_score = torch.sum(att_rep * item_feature, dim=-1)
        att_item_score = self.tanh(att_item_score)
        att_item_score = torch.sum(att_item_score, dim=0)
        return att_item_score

    def get_pos_neg_att_list_item_score(self, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask, item):
        pos_score = self.get_att_list_item_score(pos_att_list, pos_att_list_mask, item)
        neg_score = self.get_att_list_item_score(neg_att_list, neg_att_list_mask, item)
        all_score = pos_score - neg_score
        return all_score

    def get_pos_neg_att_list_item_list_score(self, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask, item_list):
        pos_score = self.get_att_list_item_list_score(pos_att_list, pos_att_list_mask, item_list)
        neg_score = self.get_att_list_item_list_score(neg_att_list, neg_att_list_mask, item_list)
        all_score = pos_score - neg_score
        return all_score

    def get_pos_neg_att_list_att_score(self, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask, att):
        pos_score = self.get_att_list_att_score(pos_att_list, pos_att_list_mask, att)
        neg_score = self.get_att_list_att_score(neg_att_list, neg_att_list_mask, att)
        all_score = pos_score - neg_score
        return all_score

    def get_pos_neg_att_list_att_list_score(self, pos_att_list, pos_att_list_mask, neg_att_list, neg_att_list_mask, att_list):
        pos_score = self.get_att_list_att_list_score(pos_att_list, pos_att_list_mask, att_list)
        neg_score = self.get_att_list_att_list_score(neg_att_list, neg_att_list_mask, att_list)
        all_score = pos_score - neg_score
        return all_score

    def get_pos_neg_att_list_item_score_single(self, pos_att_list, neg_att_list, item_list):
        pos_score = self.get_att_list_item_score_single(pos_att_list, item_list)
        neg_score = torch.zeros_like(pos_score)
        if len(neg_att_list) > 0:
            neg_score = self.get_att_list_item_score_single(neg_att_list, item_list)
        all_score = pos_score - neg_score
        return all_score

    def get_item_all_att_score(self, item, all_att):
        att_rep = self.graph_rep[self.attribute_graph_index[all_att]]
        item_feature = self.graph_rep[self.item_graph_index[item]]
        item_feature = item_feature.unsqueeze(-2)
        item_all_att_score = torch.sum(att_rep * item_feature, dim=-1)
        item_all_att_score = self.tanh(item_all_att_score)
        return item_all_att_score
