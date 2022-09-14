import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentEARConfig():
    def __init__(self, config):
    	self.use_gpu = True
    	self.input_dim = config.input_dim
    	self.hidden_dim = config.hidden_dim
    	self.output_dim = config.output_dim
    	self.dp = 0.2
    	self.DPN_model_path = root_path + "/agents/agent_ear/" + config.name + "/" #_good
    	self.DPN_model_name = "TwoLayer"
    	self.PG_discount_rate = 0.7