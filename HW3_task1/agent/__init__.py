from runner.abstracts import Agent
from . import dqn

class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
    	self.model = dqn.get_model()
    
    def step(self, state, *args, **kwargs):
        return self.model.act(state)

def create_agent(test_case_id, *args, **kwargs):
    return DQNAgent()