import random


class PIE:
    """ Implements Random Policy """

    def __init__(self, nos_actions):
        self.A = nos_actions
        self.Q = None
        self.train_count=0
        
    def predict(self, state):
        return random.randint(0, self.A-1)

    def qvals(self, state):
        qvals = [0 for _ in range(self.A)]
        return qvals
        
    def learn(self, memory, batch):
        self.train_count+=1
        return
    
    def clear(self):
        self.Q = None
        self.train_count=0
        return
        
    def render(self, mode=0):
        return "=-=-=-=-==-=-=-=-=\n RANDOM POLICY \n=-=-=-=-==-=-=-=-="
    
    def save(self, path):
        return
        
    def load(self, path):
        return
        
#-------------------