import numpy as np
#-------------------

class RND:
    def __init__(self, acts, seed=None):
        self.acts = acts
        self.train_count=0
        self.random = np.random.default_rng(seed)
        
    def predict(self, state):
        return self.random.integers(0, self.acts, size=1)[0]
        
    def learn(self, batch):
        print('Learning:', len(batch), 'transitions')
        self.train_count+=len(batch)
        return
    
    def clear(self):
        self.train_count=0
        return
        
    def render(self):
        print("=-=-=-=-==-=-=-=-=\n RANDOM POLICY \n=-=-=-=-==-=-=-=-=")
        return
    
    def save(self, path):
        print("=-=-=-=-==-=-=-=-=\n Save@",path," \n=-=-=-=-==-=-=-=-=")
        return
        
    def load(self, path):
        print("=-=-=-=-==-=-=-=-=\n Load@",path," \n=-=-=-=-==-=-=-=-=")
        return