import random

class PIE:
    """ Implements Random Policy """

    def __init__(self, nos_actions, p=print):
        self.p=p
        self.A = nos_actions
        self.Q = None
        self.train_count=0
        
    def predict(self, state):
        return random.randint(0, self.A-1)

    def qvals(self, state):
        qvals = [0 for _ in range(self.A)]
        return qvals
        
    def learn(self, memory, batch_size):
        self.p('Learning:',  batch_size,' of ', 
                            memory.count, 'transitions ... updated', self.Q )
        self.train_count+=1
        return
    
    def clear(self):
        self.Q = None
        self.train_count=0
        return
        
    def render(self, mode=0):
        self.p("=-=-=-=-==-=-=-=-=\n RANDOM POLICY \n=-=-=-=-==-=-=-=-=")
        return
    
    def save(self, path):
        self.p("=-=-=-=-==-=-=-=-=\n Save@",path," \n=-=-=-=-==-=-=-=-=")
        return
        
    def load(self, path):
        self.p("=-=-=-=-==-=-=-=-=\n Load@",path," \n=-=-=-=-==-=-=-=-=")
        return
        
#-------------------