import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class DialogueManagerConfig():
    def __init__(self, config):
        self.config = config
        self.index2attribute_path = root_path + "/data/index2attribute.json"
        self.index2item_path = root_path + "/data/index2item.json"
        self.attribute_tree_path = root_path + "/data/yelp_star_attribute_tree_dict.json"
        self.rec_action_index = 2
        self.rec_with_att_feedback_action_index = 1
        self.ask_action_index = 0
        self.rec_success_reward = 1.
        self.pos_attribute_reward = 0.1
        self.neg_attribute_reward = 0.03
        self.neg_feedback_attribute_reward = 0.03
        self.user_quit_reward = - 0.3
        self.every_turn_reward = - 0.1
        self.turn_limit = 15
