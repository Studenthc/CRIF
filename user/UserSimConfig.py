import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class UserSimConfig():
    def __init__(self, config):
        self.item_info = config.item_info
        self.att_tree_dict = config.att_tree_dict