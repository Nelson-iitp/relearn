import numpy as np

class EXP:
    """ An explorer that interacts with the environment
        Args:
            decayF is the decay function defined by user
            decayF = lambda current_epsilon, episode, timestep: new_epsilon
    """
    def __init__(self, env, mem_cap, epsilon_start, epsilon_min, epsilon_max, decayF, seed=None):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_max=epsilon_max
        self.decayF=decayF
        self.mem = MEM(capacity=mem_cap)
        self.tmem = MEM(capacity=np.inf)
        self.random = np.random.default_rng(seed)
        self.reset()
        
    def reset(self, clear_mem=True):
        self.epsilon=self.epsilon_start
        self.ts, self.es = 0, 0
        self.cS = self.env.reset()
        self.done = False
        if clear_mem:
            self.mem.clear()
            self.tmem.clear()
        
    def decay(self):
        new_epsilon=self.decayF(self.epsilon, self.es, self.ts)
        self.epsilon = min(max(self.epsilon_min, new_epsilon), self.epsilon_max)
        return

    def explore(self, pie, steps):
        """ Explore the enviroment for fixed number of 'time-steps'  """
        for k in range(steps):
            act, agreedy = (pie.predict(self.cS), 1) if (self.random.random(size=1)[0] < self.epsilon) else (self.env.action_space.sample(), 0)
            cS = self.cS
            nS, reward, self.done, _ = self.env.step(act)
            self.mem.commit(cS, nS, act, reward, self.done, (self.es, self.ts, agreedy, self.epsilon) ) # cS, nS, A, R, D
            self.decay()

            if self.done: 
                self.es+=1
                self.ts=0
                self.cS = self.env.reset() #<----- reset in terminal state
                self.done=False
            else:
                self.ts+=1
                self.cS = nS
        return
    
    def test(self, pie, steps=np.inf, reset_env=False):
        """ Explore the enviroment for fixed number of 'time-steps' with greedy mode """
        
        if reset_env:
            if self.ts!=0: # means no need to increase es
                self.ts = 0
                self.es += 1
            self.cS = self.env.reset()
            self.done=False
        ts = 0
        self.tmem.clear()
        total_reward=0
        while not self.done and ts<steps:
            ts+=1
            act = pie.predict(self.cS) 
            cS = self.cS
            nS, reward, self.done, _ = self.env.step(act)
            total_reward += reward
            self.tmem.commit(cS, nS, act, reward, self.done, (self.es, self.ts) ) 
            self.cS = nS
            self.ts+=1
            
        if self.done: 
            self.es+= 1
            self.ts = 0
            self.cS = self.env.reset() #<----- reset in terminal state
            self.done=False
        return ts, total_reward
        
#-------------------
    def render(self):
        print("=-=-=-=-==-=-=-=-=\nEXPLORER [ TS: "+str(self.ts)+" | ES: "+str(self.es)+ "]\n=-=-=-=-==-=-=-=-=")
        print('-------------------')
        print(' done:\t[', self.done, ']')
        print(' state:\t',self.cS)
        print(' mem:\t',self.mem.count,'/',self.mem.capacity)
        print(' eps:\t',self.epsilon, '{',self.epsilon_start,self.epsilon_min,self.epsilon_max ,'}')
        print('=-=-=-=-==-=-=-=-=!EXPLORER=-=-=-=-==-=-=-=-=')
        return
#-------------------

class MEM:

    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        # list-based memory
        self.mem = []
        self.count = 0
        self.cols = ("cS", "nS", "A", "R", "D", "T")
        self.random = np.random.default_rng(seed)
    
    def clear(self):
        self.mem.clear()
        self.count=0
        
    def commit(self, cS, nS, A, R, D, T): 
        """ commit a transition to memmory in this format ~ cS, nS, A, R, D, T
            cS, nS, A, R  are vectors from spaces
            D is boolean/integer
            T is tag - an object
        """
        transition=(cS, nS, A, R, D, T) # store transition as a tuple
        if self.count>=self.capacity:
            _=self.mem.pop(0)
        else:
            self.count+=1
        self.mem.append(transition)

    def sample(self, batch_size, index_only=False):
        batch_pick_size = min(batch_size, self.count)
        iSamp = self.random.integers(0, self.count, size=batch_pick_size)
        if index_only:
            batch = iSamp
        else:
            batch = self.read(iSamp)
        return batch
        
    def read(self, iSamp):
        return [ self.mem[i] for i in iSamp ]
 
    def recent(self, batch_size, index_only=False):
        batch_pick_size = min(batch_size, self.count)
        return self.mem[-batch_pick_size:]
    
    def render(self, low, high, step=1):
        print("MEMORY ["+str(self.count)+" | "+str(self.capacity)+ "]")
        print('-------------------')
        
        for rr in range (low, high, step): #(0,self.R)
            print('SLOT {}'.format(rr))
            print('\tcS:', self.mem[rr][0])
            print('\tnS:', self.mem[rr][1])
            print('\tA:', self.mem[rr][2])
            print('\tR:', self.mem[rr][3])
            print('\tD:', self.mem[rr][4])
            print('\tT:', self.mem[rr][5])
            print('-------------------')


    def render_samples(prefix, samples, cols=None):
        # self.cols = ("cS", "nS", "A", "R", "D", "T")
        if type(cols)!=type(None):
            print(prefix, type(samples), len(samples), cols)
            for j in range(len(samples)):
                print('*',j, type(samples[j]),  len(samples[j]))
                for i in range(len(samples[j])):
                    print('\t>', i, cols[i], samples[j][i])
        else:
            print(prefix, type(samples), len(samples), )
            for j in range(len(samples)):
                print('*',j, type(samples[j]),  len(samples[j]))
                for i in range(len(samples[j])):
                    print('\t>', i, samples[j][i])
#-------------------
