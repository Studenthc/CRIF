import math
import os

from utils.utils import set_random_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import random
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from utils.LogPrint import Logger
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from agents.AgentEAR import AgentEAR
from agents.AgentEARConfig import AgentEARConfig
from recommendersystem.recsys import recsys
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig
from utils.global_variable import *
from utils.config_yelp import ConfigYelp
from utils.config_last_fm import ConfigLastFM
from utils.config_last_fm_star import ConfigLastFMStar
from utils.config_yelp_star import ConfigYelpStar
from utils import user_att_count
from utils import user_att_count_not_subset
from utils import user_att_count_yelp
from utils import utils
from utils.global_variable import *

eps = np.finfo(np.float32).eps.item()
config = None


def agent_eval(agent, load_model_type, dm):
    use_gpu = True
    env = dm
    data_name = args.data_name
    if data_name == YELP_STAR:
        data_name = YELP

    PG_test_data_path = "./data/" + data_name + '/' + UI_INTERACTION_DATA + '/review_dict_test.json'

    with open(PG_test_data_path, "rb") as f:
        PG_test_data_list = json.load(f)

    ui_test_list = []
    for user_str, items in PG_test_data_list.items():
        user_id = int(user_str)
        for item_id in items:
            ui_test_list.append([user_id, item_id])
    ui_test_array = np.array(ui_test_list)
    np.random.shuffle(ui_test_array)

    test_size = 4000
    if args.data_name in [LAST_FM_STAR, LAST_FM]:
        test_size = 4000  # Only do 4000 iteration for the sake of time
    if args.data_name in [YELP_STAR, YELP]:
        test_size = 2500  # Only do 2500 iteration for the sake of time

    agent.set_env(env)
    if load_model_type == "PG":
        print("load PG model ...")
        agent.load_model()
    else:
        print("no pretrian model...")

    sum_reward = 0.
    sum_turn = 0
    sum_success = 0
    episode_num = 0
    SR_turn_15 = [0] * 15
    Rank = 0
    for e_data in tqdm(random.sample(ui_test_list, test_size), ncols=0):
        # user = ui_test_array[episode_num, 0]
        user = e_data[0]
        # item = ui_test_array[episode_num, 1]
        item = e_data[1]
        reward, turn, success, state_list = agent.PG_eva_one_episode(user, item)
        if 4 >= turn >= 2:
            contain_rec_feed = False
            contain_ask_suc = False
            for state in state_list:
                if state[0] == '4':
                    contain_rec_feed = True
                if state[0] == '1':
                    contain_ask_suc = True
            if state_list[-1][0] == '2' and contain_rec_feed:
                print("get one")
                print(e_data[0], e_data[1])
                print(len(state_list))
                print(state_list)
                if turn <= 3:
                    print("get better one")
                    print(e_data[0], e_data[1])
                    print(len(state_list))
                    print(state_list)
        episode_num += 1
        sum_reward += reward
        sum_turn += turn
        sum_success += success

        if success:
            t = turn - 1
            SR_turn_15 = [v + 1 if i > t else v for i, v in enumerate(SR_turn_15)]
            Rank += (1 / math.log(t + 3, 2) + (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) / math.log(True + 1, 2))

    average_reward = float(sum_reward) / episode_num
    average_turn = float(sum_turn) / episode_num
    success_rate = float(sum_success) / episode_num
    SR_TURN = [i / test_size for i in SR_turn_15]
    print(SR_TURN)
    print(Rank / test_size)

    time_str = datetime.datetime.now().isoformat()
    print("{}: average_reward {:g}, average_turn {:g}, success_rate {:g}"\
            .format(time_str, average_reward, average_turn, success_rate))

parser = argparse.ArgumentParser(description='train ear agent')
parser.add_argument('--mode', type=str,
                    help='choose from pretrain or PG')
parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                    help='One of {lastfm, lastfm_star, yelp, yelp_star}.')
args = parser.parse_args()

if args.data_name == LAST_FM:
    config = ConfigLastFM()
    user_att_count_not_subset.set_config(config)
    model_name = "iter299-2022-09-01" # lastfm
elif args.data_name == YELP:
    config = ConfigYelp()
    user_att_count_not_subset.set_config(config)
    model_name = "iter800-2022-09-06" # yelp 2loss
elif args.data_name == LAST_FM_STAR:
    config = ConfigLastFMStar()
    user_att_count_not_subset.set_config(config)
    model_name = "iter4260-2022-09-01" # lastfm_star
elif args.data_name == YELP_STAR:
    config = ConfigYelpStar()
    user_att_count_not_subset.set_config(config)
    model_name = "iter800-2022-09-03" # yelp* 3loss equal

ch = ConvHis(ConvHisConfig(config))
agent = AgentEAR(AgentEARConfig(config), ch)
rec = recsys(config, convhis=ch)
ch.set_rec(rec)
rec.load_model(model_name, False)
rec.init_eval(config.adj_index)
usersim = UserSim(UserSimConfig(config))
dm = DialogueManager(DialogueManagerConfig(config), rec, agent, usersim, ch)
set_random_seed(1)
agent_eval(agent, "PG", dm)
