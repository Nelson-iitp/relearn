import numpy as np
import json

class PIE:
    """ Implements Dictionary-Based Q-Learning 
    
        Q(s,a) = (1-lr)*Q(s,a) + lr*(R(s,a,s') + dis*maxQ(s',A)) 
        
    
        Initialize new Q-Learner on discrete action space 
        
        Args:
            lr              learning rate  
            ls              learn steps (usually 1)
            dis             discount factor 
            nos_actions     no of discrete actions (0 to acts-1)
            mapper          ptional { function: state(array) --> state(string) }
                            .. mapper is a function that returns the string representation 
                            .. of a state vector to be stored in dict keys 

        # target = reward + self.dis * max(self.Q[nS][0]) * int(not done) 
        # Note: we should store Q-Values as 
        #  { 'si' : [ Q(si,a1), Q(si,a2), Q(si,a3), ... ] }
        # but we also want to store the number of times a state was visited
        # instaed of creating seperate dictionary, we use same dict and store #visited in position
        #  { 'si' : [ [ Q(si,a1), Q(si,a2), Q(si,a3), ... ], #visited ] }
        """
        
    def __init__(self, nos_actions, lr=0.5, dis=1, mapper=str):
        self.lr, self.dis = lr, dis           
        self.A = nos_actions
        self.Q={}                 # the Q-dictionary where Q-values are stored
        self.mapper = mapper      # a function that mapps state vectors to strings
        self.train_count = 0      # counts the number of updates made to Q-values
        self.random = np.random.default_rng()
        
    def predict(self, state):
        cS = self.mapper(state)
        if not cS in self.Q:
            action = self.random.integers(0, self.A, size=1)[0]
        else:
            qvals = (self.Q[cS][0])
            qis = np.where(qvals==np.max(qvals))[0]
            action = self.random.choice(qis, size=1)[0]
        return action
        
    def qvals(self, state):
        cS = self.mapper(state)
        if not cS in self.Q:
            qvals = [0 for _ in range(self.A)] #  [0 for _ in range(self.A)]  ,  np.zeros(self.A)
        else:
            qvals = self.Q[cS][0]
        return qvals
        
    def learn(self, memory, batch):
        """ memory = explorer's memory
            batch = indices in memomry, use following
                    batch = memory.recent(batch_size)
                    batch = memory.sample(batch_size)
                    batch = memory.all()
        """
        # batch = memory.recent(batch_size) # sample most recent 
        for i in batch:
            cS, nS, act, reward, done = memory.mem[i] 
            cS, nS = self.mapper(cS), self.mapper(nS)
            if not cS in self.Q:
                self.Q[cS] = [[0 for _ in range(self.A)], 1]
            else:
                self.Q[cS][1]+= 1
            if not nS in self.Q: 
                self.Q[nS] = [[0 for _ in range(self.A)], 0]
                
            self.Q[cS][0][act] = (1-self.lr)    *   ( self.Q[cS][0][act] ) + \
                                 (self.lr)      *   ( reward + self.dis * np.max(self.Q[nS][0]) * int(not done) )  #<--- Q-update
            self.train_count+=1
        return 

    def clear(self):
        self.Q.clear()
        self.train_count=0
        return

    def render(self, mode=0):
        """ use mode=1 to view full dictionary """
        res='=-=-=-=-==-=-=-=-=\nDICT: Q-Values  #'+str(len(self.Q))+'\n=-=-=-=-==-=-=-=-=\n'
        if mode>0:
            for i in self.Q:
                res+=str(i) + '\t\t' + str(self.Q[i]) + '\n'
            res = res + '=-=-=-=-==-=-=-=-='
        return res
        
    def save(self, path):
        f=open(path, 'w')
        f.write(json.dumps(self.Q, sort_keys=False, indent=4))
        f.close()
        return
        
    def load(self, path):
        f=open(path, 'r')
        self.Q = json.loads(f.read())
        f.close()
        return

#-------------------