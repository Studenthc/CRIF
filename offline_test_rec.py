import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import argparse
from recommendersystem.recsys import recsys
from utils.build_test_rec_data_loader_new import *
from utils.global_variable import *
from utils.config_yelp import ConfigYelp
from utils.config_last_fm import ConfigLastFM
from utils.config_last_fm_star import ConfigLastFMStar
from utils.config_yelp_star import ConfigYelpStar
from utils import utils
from utils import user_att_count
from utils import user_att_count_yelp

use_gpu = True
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                    help='One of {lastfm, lastfm_star, yelp, yelp_star}.')
args = parser.parse_args()
config = None
if args.data_name == LAST_FM:
    config = ConfigLastFM()
    user_att_count.set_config(config)
    model_name = "iter299-2022-09-01" # lastfm
elif args.data_name == YELP:
    config = ConfigYelp()
    user_att_count_yelp.set_config(config)
    model_name = "iter800-2022-09-06" # yelp
elif args.data_name == LAST_FM_STAR:
    config = ConfigLastFMStar()
    user_att_count.set_config(config)
    model_name = "iter4260-2022-09-01" # lastfm_star
elif args.data_name == YELP_STAR:
    config = ConfigYelpStar()
    user_att_count.set_config(config)
    model_name = "iter800-2022-09-03" # yelp_star
rec = recsys(config, use_gpu=use_gpu)

data_name = args.data_name
if data_name == YELP_STAR:
    data_name = YELP
test_info_path = "./data/" + data_name + '/' + FM_SAMPLE_DATA + "/sample_fm_data_valid.pkl"
test_info = utils.load_pickle(test_info_path)


def rec_test(test_model_path=None):
    if test_model_path is not None:
        if use_gpu:
            rec.rec.load_state_dict(torch.load(test_model_path))
        else:
            rec.rec.load_state_dict(torch.load(test_model_path, map_location='cpu'))
    rec.init_eval(config.adj_index)

    test_batch_size = 128
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
        attribute_score = rec.rec.get_att_list_score(user_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask,
                                                     test_attribute_list)
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

    mean_att_auc = np.mean(np.array(att_auc_list))
    mean_item_auc = np.mean(np.array(item_auc_list))
    mean_global_item_auc = np.mean(np.array(global_item_auc_list))
    mean_item_att_auc = np.mean(np.array(item_att_auc_list))

    print("---------test-----------")
    print("item_auc: {}, att_auc: {}, global item_auc: {}, item_att_auc: {}".format(str(mean_item_auc),
                                                                       str(mean_att_auc),
                                                                        str(mean_global_item_auc),
                                                                       str(mean_item_att_auc)))
    return mean_item_auc, mean_att_auc, mean_global_item_auc, mean_item_att_auc

rec.load_model(model_name, False)
rec.init_eval(config.adj_index)
with torch.no_grad():
    rec_test()
