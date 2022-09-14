import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class ConvHisConfig():
    def __init__(self, config):
        self.config = config
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.att_tree_dict = config.att_tree_dict
        self.user_info = config.user_info
        self.item_info = config.item_info
        self.att_pos_state = 1
        self.att_neg_state = -1
        self.item_neg_state = -2
        self.init_state = 0
        self.max_conv_length = 15
        self.user_info_path = root_path + "/data/user_info.json"
        self.item_info_path = root_path + "/data/item_info.json"
        self.attribute_tree_path = root_path + "/data/yelp_star_attribute_tree_dict.json"