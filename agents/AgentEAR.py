import torch
import datetime
from torch.distributions import Categorical
from agents.DeepPolicyNetwork import TwoLayersModel
from agents.RewardNetwork import RewardModel
from agents.RecommendFailed import RecommendFailed
from agents.RewardJudgeSingleAction import human_feedback

class AgentEAR:
    def __init__(self, config, convhis):
        self.convhis = convhis
        self.use_gpu = config.use_gpu
        self.DPN = TwoLayersModel(config)
        self.RWN = RewardModel(config)
        self.DPN_model_path = config.DPN_model_path
        self.DPN_model_name = config.DPN_model_name
        self.aciton_len = config.output_dim

        self.rec = None
        self.env = None

    def set_rec_model(self, rec_model):
        self.rec = rec_model

    def set_env(self, env):
        self.env = env

    def get_reward(self, state):
        return self.RWN(state.cuda())

    def init_episode(self):
        self.DPN.eval()
        self.RWN.eval()

    def save_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        time_str = datetime.datetime.now().isoformat()
        torch.save(self.DPN.state_dict(), "".join([self.DPN_model_path, self.DPN_model_name + name_suffix + time_str]))
        torch.save(self.RWN.state_dict(), "".join([self.DPN_model_path, self.DPN_model_name + "_RWN" + time_str]))

    def load_model(self):
        name_suffix = "_PG"
        self.DPN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + name_suffix])))
        self.RWN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + "_RWN"])))

    def PG_train_one_episode(self, user, item):
        self.DPN.train()
        self.RWN.train()
        state_pool = []
        action_pool = []
        reward_pool = []
        reward_train_list = []

        state = self.env.initialize_episode(user, item)
        IsOver = False
        success = False
        while not IsOver:
            attribute_distribution = self.DPN(state.float().cuda(), True)
            c = Categorical(probs = attribute_distribution)

            action = c.sample()
            
            IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
            feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state = self.env.step(action)
            label_tensor = torch.tensor([0, 1])
            ask_reward_more = human_feedback(be_len, be_rank, ask_len, ask_rank, feedback_rec_len, feedback_rec_rank)
            if ask_reward_more:
                label_tensor = torch.tensor([1, 0])
            reward_train_list.append((ask_reward, feedback_rec_reward, label_tensor))

            state_pool.append(state)
            action_pool.append(c.log_prob(action))
            reward_pool.append(reward.item())

            if not IsOver:
                state = next_state

        return action_pool, reward_pool, success, reward_train_list

    def PG_eva_one_episode(self, user, item, silence=True):
        self.DPN.eval()
        self.RWN.eval()
        total_reward = 0.
        turn_count = 0
        is_success = False
        state_list = []

        state = self.env.initialize_episode(user, item)
        IsOver = False
        while not IsOver:
            turn_count += 1
            attribute_distribution = self.DPN(state.float().cuda(), True)

            action = int(attribute_distribution.argmax())

            IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
            feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state = self.env.step(action)
            state_list.append(step_state)
            total_reward += reward
            is_success = success
            if not IsOver:
                state = next_state

        return total_reward, turn_count, is_success, state_list
