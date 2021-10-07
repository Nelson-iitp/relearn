import numpy as np
import json
#-------------------

class TQL:
    def __init__(self, acts, theta, dis, mapper=str, seed=None):
        """ Initialize new Q-Learner 
        
        Args:
            theta          learning rate   (theta)
            dis         discount factor (gamma)
            acts        no of discrete actions (0 to acts-1)
            mapper      optional { function: state(array) --> state(string) }
                            .. mapper is a function that returns the string representation 
                            .. of a state vector to be stored in dict keys 
            rseed       seed for numpy PRNG
        
        """
        # Note: we store Q-Values as 
        #  { 'si' : [ Q(si,a1), Q(si,a2), Q(si,a3), ... ] }
        # we also want to store the number of times a state was visited
        # instaed of creating seperate dictionary, we use same dict and store #visited in position
        #  { 'si' : [ [ Q(si,a1), Q(si,a2), Q(si,a3), ... ], #visited ] }
        self.theta, self.dis, self.acts = theta, dis, acts              
        self.mapper = mapper  
        self.Q={}                 # the Q-dictionary where Q-values are stored
        self.train_count = 0      # counts the number of updates made to Q-values
        self.random = np.random.default_rng(seed)

    def predict(self, state):
        cS = self.mapper(state)
        if not cS in self.Q:
            action = self.random.integers(0, self.acts, size=1)[0]
        else:
            qvals = self.Q[cS][0]
            action = self.random.choice(np.where(qvals==max(qvals))[0], size=1)[0]
        return action

    def learn(self, batch):
        steps = len(batch)
        #new_states = 0 # record the number of new states discovered in this batch
        for k in range(steps):
            cS, nS, act, reward, done, tag = batch[k] 
            
            cS = self.mapper(cS)
            if not cS in self.Q:
                self.Q[cS] = [[0 for _ in range(self.acts)], 1]
                #new_states+=1
            else:
                self.Q[cS][1]+= 1
            
            nS = self.mapper(nS)
            if not nS in self.Q: 
                self.Q[nS] = [[0 for _ in range(self.acts)], 1]
                #new_states+=1
            else:
                self.Q[nS][1]+= 1
            
            target = reward + self.dis * max(self.Q[nS][0]) * int(not done) 
            self.Q[cS][0][act] = (1-self.theta) * self.Q[cS][0][act] + (self.theta) * target  #<--- Q-update
            self.train_count+=1
        # End Iteration (steps)
        return 

    def clear(self):
        self.Q.clear()
        self.train_count=0
        return

    def render(self):
        """ returns a string representation of a dict object for printing """
        res='=-=-=-=-==-=-=-=-=\nDICT: Q-Values\n=-=-=-=-==-=-=-=-=\n'
        for i in self.Q:
            res+=str(i) + '\t\t' + str(self.Q[i]) + '\n'
        res = res + '=-=-=-=-==-=-=-=-='
        print(res)
        return
        
    def save(self, path):
        print("=-=-=-=-==-=-=-=-=\n Save@",path," \n=-=-=-=-==-=-=-=-=")
        f=open(path, 'w')
        f.write(json.dumps(self.Q, sort_keys=False, indent=4))
        f.close()
        return
        
    def load(self, path):
        print("=-=-=-=-==-=-=-=-=\n Load@",path," \n=-=-=-=-==-=-=-=-=")
        f=open(path, 'r')
        self.Q = json.loads(f.read())
        f.close()
        return
#-------------------


#-------------------
#-------------------
#-------------------