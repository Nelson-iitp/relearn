import random
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------------------
# Defines EXP and MEM
#---------------------------------------------------------


#---------------------------------------------------------

class EXP:
    """ An explorer that interacts with the environment
        with greedy-epsilon policy i.e epsilon = exploration prob 
        
        Args:
            env                 base enviroment for exploration
            cap                 max memory capacity
            epsilon             for epsilon-greedy exploration
            test                if True, does not use epsilon-greedy exploration 
                                     ... and does not store state-vectors in memory
            
        self.env must implement:
            self.env.reset()
            self.env.action_space.sample()
            self.env.step(act)
            self.env._elapsed_steps
            self.env._max_episode_steps
    """
    
    def __init__(self, env, cap, epsilon, test=False):
        self.memory = MEM(capacity=cap, test=test)
        self.random = np.random.default_rng()
        self.epsilon_min, self.epsilon_max = epsilon
        self.epsilon = epsilon[0] # initially set min as start value
        self.test=test
        if self.test:
            self.step=self.stepT
        else:
            self.step=self.stepE
        self.env = env
        self.reset()  

    def set_seed(self, seed):
        self.random = np.random.default_rng(seed)
        
    def reset(self):
        self.cS, self.done = self.env.reset(), False
        
    
    def stepE(self, pie):
        act = self.env.action_space.sample() \
              if (self.random.random(size=1)[0] < self.epsilon) \
              else pie.predict(self.cS)
        cS = self.cS 
        nS, reward, self.done, _ = self.env.step(act)
        transition = (cS, nS, act, reward, self.done) #<<--- as per MEM.DEFAULT_SCHEMA
        self.memory.commit(transition)
        if self.done or self.env._elapsed_steps>=self.env._max_episode_steps: 
            self.reset()
            self.memory.mark()
            done=True
        else:
            self.cS = nS
            done=False
        return done
        
    def stepT(self, pie):
        act = pie.predict(self.cS)
        self.cS, reward, self.done, _ = self.env.step(act)
        transition = (act, reward, self.done) 
        self.memory.commit(transition)
        if self.done or self.env._elapsed_steps>=self.env._max_episode_steps: 
            self.reset()  
            self.memory.mark()
            done=True
        else:
            done=False
        return done
        
    def episode(self, pie):
        done, ts = False, 0
        while not done:
            done = self.step(pie)
            ts+=1
        return ts
        
    def explore(self, pie, moves, decay, episodic=False):
        if self.test:
            if episodic:
                ts = 0
                for k in range(moves):
                    ts += self.episode(pie)
            else:
                ts = moves
                for k in range(moves):
                    done = self.step(pie)
        else:
            if episodic:
                ts = 0
                for k in range(moves):
                    ts += self.episode(pie)
                    self.epsilon = min(max( decay(self.epsilon, (k+1)/moves, ts, True), self.epsilon_min ), self.epsilon_max) 
            else:
                ts = moves
                for k in range(moves):
                    done = self.step(pie)
                    self.epsilon = min(max( decay(self.epsilon, (k+1)/moves, k+1, done), self.epsilon_min ), self.epsilon_max) 
        return ts
    
    def summary(self):
        # npe = [action, reward, done]
        if self.test:
            npe = np.array(self.memory.read_cols(self.memory.all(), 0, 3 ))  #np.array(self.memory.mem)
        else:
            npe = np.array(self.memory.read_cols(self.memory.all(), 2, 5 ))
            
        clean_up=False
        # assume that memory.mark is corrrectly called()
        if len(self.memory.episodes)==0:
            self.memory.episodes.append(self.memory.count)
            clean_up=True
        else:
            if self.memory.episodes[-1]!=self.memory.count:
                self.memory.episodes.append(self.memory.count)
                clean_up=True
        
        si = 0
        cnt = 0
        header = np.array([' #','Start','End', 'Steps','Reward','Done'])
        rows= []
        for ei in self.memory.episodes:
            cnt+=1
            ep = npe[si:ei] # ep = [action, reward, done]
            aseq, rsum = ep[:,0], np.sum(ep[:,1]) # action sequence, total reward
            row = []
            rows.append([ cnt, si, ei, len(aseq), rsum, int(ep[-1,2]) ])
            si = ei
        if clean_up:
            del self.memory.episodes[-1]
            
        rows = np.array(rows)
        avg_reward =  np.mean(rows[:, 4])
        return header, rows, avg_reward
#---------------------------------------------------------

class MEM:
    """ A list based memory for explorer """

    def __init__(self, capacity, test=False, p=print):
        self.capacity = capacity
        self.mem = []
        self.episodes=[]
        self.count = 0
        self.test = test
        if self.test:
            self.render=self.renderT
        else:
            self.render=self.renderE
        self.random = np.random.default_rng()
        
        if self.test:
            self.render_schema = ('Action', 'Reward', 'Done')
        else:
            self.render_schema = ('cS', 'nS', 'Action', 'Reward',  'Done')
            
    
    def set_seed(self, seed):
        self.random = np.random.default_rng(seed)
    
    def clear(self):
        self.mem.clear()
        self.episodes.clear()
        self.count=0
        return
    def commit(self, transition): 
        if self.count>=self.capacity:
           del self.mem[0:1]
        else:
            self.count+=1
        self.mem.append(transition)
        return

    def mark(self):
        self.episodes.append(self.count)
        return
        
    def sample(self, batch_size):
        batch_pick_size = min(batch_size, self.count)
        return self.random.integers(0, self.count, size=batch_pick_size)
    def recent(self, batch_size):
        batch_pick_size = min(batch_size, self.count)
        return np.arange(self.count-batch_pick_size, self.count, 1)
    def all(self):
        return np.arange(0, self.count, 1)
        
    def read(self, iSamp):
        return [ self.mem[i] for i in iSamp ]
    def read_col(self, iSamp, iCol):
        return [ self.mem[i][iCol] for i in iSamp ]
    def read_cols(self, iSamp, iCol_from, iCol_to):
        return [ self.mem[i][iCol_from:iCol_to] for i in iSamp ]

    def renderE(self, low, high, step=1, p=print):
        
        p('=-=-=-=-==-=-=-=-=@MEMORY=-=-=-=-==-=-=-=-=')
        p("Status ["+str(self.count)+" | "+str(self.capacity)+ "]")
        p('------------------@SLOTS------------------')
        
        for i in range (low, high, step):
            p('SLOT: [', i, ']')
            for j in range(len(self.mem[i])):
                p('\t',self.render_schema[j],':', self.mem[i][j])
            p('-------------------')
        p('=-=-=-=-==-=-=-=-=!MEMORY=-=-=-=-==-=-=-=-=')
        return     
    def renderT(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@MEMORY=-=-=-=-==-=-=-=-=')
        p("Status ["+str(self.count)+" | "+str(self.capacity)+ "]")
        
        res = 'SLOT \t'
        for j in range(len(self.render_schema)):
                res+= (self.render_schema[j]+'\t')
        p(res)
        
        for i in range (low, high, step):
            res=str(i)+' \t'
            for j in range(len(self.mem[i])):
                res+=str(self.mem[i][j])+' \t'
            p(res)
        p('=-=-=-=-==-=-=-=-=!MEMORY=-=-=-=-==-=-=-=-=')
        return
    def render_all(self, p=print):
        self.render(0, self.count, p=p)
        return
    def render_last(self, nos, p=print):
        self.render(-1, -nos, step=-1,  p=p)
        return


#---------------------------------------------------------
