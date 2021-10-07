from relearn.pies.rnd import RND
from relearn.pies.tql import TQL
from relearn.pies.dqn import DQN

# all policies must implement  _i_PLCRSL
"""
    def __init__(self, acts, seed=None):
    def predict(self, state): return action
    def learn(self, batch): return
    def clear(self):return
    def render(self): return
    def save(self, path): return
    def load(self,path): return
"""