import numpy as np
import matplotlib.pyplot as plt
import json
#---------------------------------------------------------
# Defines EXP and MEM
# Defines two inbuilt policies - RND and TQL
#---------------------------------------------------------

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

    def sample(self, batch_size):
        """ by default, it returns indices, use read() to get actual batch """
        batch_pick_size = min(batch_size, self.count)
        return self.random.integers(0, self.count, size=batch_pick_size)
        # batch = self.read(sample(self, batch_size))
    def recent(self, batch_size):
        batch_pick_size = min(batch_size, self.count)
        return np.arange(self.count-batch_pick_size, self.count, 1)
    def read(self, iSamp):
        return [ self.mem[i] for i in iSamp ]

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


class RND:
    def __init__(self, nos_actions, seed=None):
        """ Implements Random Policy
        """
        self.A = nos_actions
        self.Q = None
        self.train_count=0
        self.random = np.random.default_rng(seed)
        
    def predict(self, state):
        return self.random.integers(0, self.A, size=1)[0]

    def qvals(self, state):
        qvals = [0 for _ in range(self.A)]
        return qvals
        
    def learn(self, memory, batch_size):
        print('Learning:',  batch_size,' of ', 
                            memory.count, 'transitions ... updated', self.Q )
        self.train_count+=1
        return
    
    def clear(self):
        self.Q = None
        self.train_count=0
        return
        
    def render(self, mode=0):
        print("=-=-=-=-==-=-=-=-=\n RANDOM POLICY \n=-=-=-=-==-=-=-=-=")
        return
    
    def save(self, path):
        print("=-=-=-=-==-=-=-=-=\n Save@",path," \n=-=-=-=-==-=-=-=-=")
        return
        
    def load(self, path):
        print("=-=-=-=-==-=-=-=-=\n Load@",path," \n=-=-=-=-==-=-=-=-=")
        return
        
#-------------------

class TQL:
    """ Implements Dictionary-Based Q-Learning 
    
        Q(s,a) = (1-lr)*Q(s,a) + lr*(R(s,a,s') + dis*maxQ(s',A)) 
        
    """
    def __init__(self, nos_actions, lr, dis, mapper=str, seed=None):
        """ Initialize new Q-Learner on discrete action space 
        
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
        self.lr, self.dis, self.acts = lr, dis, nos_actions              
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
        
    def qvals(self, state):
        cS = self.mapper(state)
        if not cS in self.Q:
            qvals = [0 for _ in range(self.acts)]
        else:
            qvals = self.Q[cS][0]
        return qvals
        
    def learn(self, memory, batch_size):
        batch = memory.recent(batch_size) # sample most recent 
        #steps=len(batch) # note: batch len may not always be batch_size
        for i in batch:
            cS, nS, act, reward, done, tag = memory.mem[i] 
            cS = self.mapper(cS)
            if not cS in self.Q:
                self.Q[cS] = [[0 for _ in range(self.acts)], 1]
            else:
                self.Q[cS][1]+= 1
            nS = self.mapper(nS)
            if not nS in self.Q: 
                self.Q[nS] = [[0 for _ in range(self.acts)], 0]
                
            self.Q[cS][0][act] = (1-self.lr)    *   ( self.Q[cS][0][act] ) + \
                                 (self.lr)      *   ( reward + self.dis * max(self.Q[nS][0]) * int(not done) )  #<--- Q-update
            self.train_count+=1
        return 

    def clear(self):
        self.Q.clear()
        self.train_count=0
        return

    def render(self, mode=0):
        res='=-=-=-=-==-=-=-=-=\nDICT: Q-Values  #'+str(len(self.Q))+'\n=-=-=-=-==-=-=-=-=\n'
        if mode>0:
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

def TRAIN(exp, pie, epochs, esteps, sample_size, report, texp, tepochs, tsteps, treset, verbose=True):

    if verbose:
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: INFO')
        #print('observation_space', exp.env.observation_space) # = gym.spaces.box.Box(-inf, inf, shape=(env.LEN,))
        #print('action_space', exp.env.action_space) # = gym.spaces.discrete.Discrete(env.A)
        exp.render()
        #pie.render()
        # learning loop params
        print('epochs', epochs)
        print('esteps', esteps)
        print('tsteps', tsteps)
        print('sample_size', sample_size)
        print('report', report)
        print('treset', treset)
        print('verbose', verbose)
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  !TRAINER: INFO\n')


    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: BEGIN')
    epoch, hist = 0, [] # history of test rewards ( and timesteps )
    while (epoch<epochs):
        epoch+=1
        exp.explore(pie=pie, steps=esteps)
        pie.learn(exp.mem, sample_size)
        # report 
        if epoch%report==0: # test updated policy
            stepsA, rewardA =[], []
            for ep in range (tepochs):
                steps, reward = texp.test(pie=pie, steps=tsteps, reset_env=treset)
                stepsA.append(steps)
                rewardA.append(reward)
                
                if verbose:
                    print(' --> epoch:', epoch, ' steps:', steps, ' reward:', reward )
            hist.append([np.mean(stepsA), np.mean(rewardA), exp.epsilon])
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: END\tTotal Epochs:', epoch,'\n')
    

    if verbose:
        exp.render()
        #pie.render()
        
        # plot test reward history
        fig, ax = plt.subplots(3,1,figsize=(16,10), sharex=True)
        h = np.array(hist)
        fig.suptitle('TRAIN', fontsize=12)
        
        ax[2].plot(h[:,2], linewidth=0.6, label='epsilon', color='tab:purple')
        ax[2].set_ylabel('epsilon')
        
        ax[1].plot(h[:,1], linewidth=0.6, label='reward', color='tab:green')
        ax[1].set_ylabel('reward')

        ax[0].plot(h[:,0], linewidth=0.6, label='steps', color='tab:blue')
        ax[0].set_ylabel('steps')

        plt.show()
    return



def TEST(texp, pie, tepochs, tsteps, treset=True, verbose=2, mode=0):

    if verbose>0:
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: INFO')
        #print('observation_space', texp.env.observation_space) # = gym.spaces.box.Box(-inf, inf, shape=(env.LEN,))
        #print('action_space', texp.env.action_space) # = gym.spaces.discrete.Discrete(env.A)

        # testing loop params
        print('epochs', tepochs)
        print('tsteps', tsteps)
        print('verbose', verbose)
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  !TESTER: INFO\n')
        if verbose>2:
            pie.render(mode=mode)
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: BEGIN')
    epoch, hist = 0, [] # history of test rewards ( and timesteps )
    
    while (epoch<tepochs):
        epoch+=1
        steps, reward = texp.test(pie=pie, steps=tsteps, reset_env=treset)
        hist.append([steps, reward])
        if verbose>1:
            print(' --> epoch:', epoch, ' steps:', steps, ' reward:', reward )
    # end of test
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: END\tTotal Epochs:', epoch,'\n')
    hist= np.array(hist)
    avg_steps = np.mean(hist[:,0])
    avg_reward = np.mean(hist[:,1])
    if verbose>0:
        print('Avg Steps:', avg_steps)
        print('Avg Reward:', avg_reward)
        #pie.render()
        # plot test reward history
        fig, ax = plt.subplots(2,1,figsize=(16,10), sharex=True)
        h = np.array(hist)
        fig.suptitle('TEST', fontsize=12)
        ax[1].plot(h[:,1], linewidth=0.6, label='reward', color='tab:blue')
        ax[1].set_ylabel('reward')

        ax[0].plot(h[:,0], linewidth=0.6, label='steps', color='tab:purple')
        ax[0].set_ylabel('steps')

        plt.show()
    return

#-------------------



#-------------------




#-------------------