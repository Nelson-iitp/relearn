
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mdp.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch as tt
import torch.distributions as td
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
from .common import int2base



def integer_reward_matrix(size, reward_range, dtype=np.float32):
    return np.random.randint(
                reward_range[0], 
                reward_range[1], 
                size=size).astype(dtype)

def real_reward_matrix(size, reward_range, dtype=np.float32):
    return  reward_range[0] + \
        (np.random.random(size=size).astype(dtype)) * (reward_range[1] - reward_range[0])
           
class treeMDP(gym.Env): # a discrete action mdp
    
    def __init__(self, n, h, reward_range=(0,1), use_integer_reward=False) -> None:
        self.n = n
        self.h = h # this starts from 0 upto h (so total height levels+1)
        self.nodes_at_level = lambda l: int(self.n**(l))
        self.nodes_upto_level = lambda l: int( (1-(self.n**l))/ (1-self.n))
        self.nodes_total = self.nodes_upto_level(self.h)
        self.nodes_max = self.nodes_at_level(self.h-1)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.S=self.observation_space.sample()*0
        self.Si=self.S[0:1]
        self.Li=self.S[1:2]

        self.action_space = gym.spaces.Discrete(self.n)
        self._max_episode_steps = self.h - 1
        self.reward_range = reward_range

        self.possible_soultions = self.n**(self.h-1)
        self.reset()

        self.size=(self.nodes_total-self.nodes_max, self.n)
        if not(use_integer_reward is None):
            self.R = (  integer_reward_matrix(self.size, self.reward_range) \
                        if use_integer_reward else \
                        real_reward_matrix(self.size, self.reward_range)
                    )
        else:
            self.R = None

    def reset(self):
        self.S*=0
        self.ts = 0 # timestep
        self.hist_l, self.hist_i =[int(self.Li)], [int(self.Si)]
        self.hist_a = [-1]
        self.hist_r = [0]
        return self.S 
        
    def step(self, action):
        # first check current level
        
        reward = self.R[int(self.nodes_upto_level(self.Li-1) + self.Si), action]
        self.Si[:] = self.n*self.Si + action
        self.Li[:]+= 1
        self.ts+=1
        self.hist_l.append(int(self.Li))
        self.hist_i.append(int(self.Si))
        self.hist_a.append(action)
        self.hist_r.append(reward)
        return self.S, reward, (self.Li>=self.h-1), {}

    def render(self):
        #s = 0
        fig = plt.figure(figsize=(12,12))
        plt.ylim(-1, self.h)
        #plt.xlim( -1, self.nodes_max+1)
        for l in range(self.h):
            n = self.nodes_at_level(l)
            delta = (self.nodes_max/2) - n/2
            x, y = np.arange(0,n,1) + delta, np.zeros(n)+l
            #print(f'{l=}, {n=}, {x=}, {y=}')
            plt.scatter(x,y)

        for l in range(1, len(self.hist_i)):
            n =  self.nodes_at_level(l)
            p =  self.nodes_at_level(l-1)
            deltap = (self.nodes_max/2) - p/2
            deltai = (self.nodes_max/2) - n/2
            plt.scatter([self.hist_i[l]+deltai], [self.hist_l[l]], color='gold', marker='s')
            plt.annotate(str(f'{self.hist_a[l]} +[{self.hist_r[l]}]'), xy=(self.hist_i[l-1]+deltap, self.hist_l[l-1]+0.2 ))

            plt.annotate("",
            xy=( self.hist_i[l]+deltai, self.hist_l[l] ),
            xytext=(self.hist_i[l-1]+deltap, self.hist_l[l-1] ), 
            arrowprops=dict(arrowstyle="->", linewidth=0.8, color='tab:pink'))
        plt.show()

    def baseline(mdp):
        # returns all possible soultion and their rewards
        # possible soultions = self.n**(self.h-1)
        from pandas import DataFrame
        hist_return, hist_steps, hist_acts = [], [], []
        for i in range(mdp.possible_soultions):
            actions = int2base(i, mdp.n, mdp.h-1)
            hist_acts.append(actions)
            _=mdp.reset()
            d, tr, ts = 0, 0, False
            while not d:
                a = actions[ts]
                ts+=1
                _, r, d, _ = mdp.step(a)
                tr+=r
                #print(f'{s=},{a=},{r=},{d=}')
            hist_return.append(tr)
            hist_steps.append(ts)
        
        test_results = DataFrame(data = {
        '#' :       range(mdp.possible_soultions),
        'steps'  :  hist_steps, 
        'return' :  hist_return, 
        'actions':  hist_acts,
        })
        return test_results
        
    def clone(self):
        res = treeMDP(self.n, self.h, self.reward_range, use_integer_reward=None )
        res.R = np.copy(self.R)
        return res

class TransitionGenerator:
    def deterministic(nos_states, near, increment):
        res = np.zeros((nos_states,), dtype=np.float32) + near
        res[np.random.randint(0, nos_states)] += increment
        return tt.tensor(res)

    def similar(nos_states, whatever1, whatever2):
        return tt.ones(nos_states)

    def uniform(nos_states, low, high):
        return tt.tensor(np.random.uniform(low, high, nos_states))

    def normal(nos_states, low, high):
        return tt.tensor(np.random.uniform(low, high, nos_states))

class randMDP(gym.Env): # a discrete action mdp
    
    def __init__(self, nos_states, nos_actions, initial_states, final_states, max_ts, reward_range=(0,1), 
    use_integer_reward=False, tr_dist=('deterministic', 0.0, 1.0), use_logits=True, seed=None) -> None:
        # a random mdp
        self.nos_states = nos_states
        self.initial_states = initial_states
        self.final_states = final_states
        self.nos_actions = nos_actions # this starts from 0 upto h (so total height levels+1)

        self.observation_space = gym.spaces.Box(low=0, high=nos_states, shape=(1,), dtype=np.int32)
        self.S=self.observation_space.sample()*0

        self.action_space = gym.spaces.Discrete(self.nos_actions)
        self._max_episode_steps = max_ts
        self.reward_range = reward_range

        self.seed=seed
        self.rng = np.random.default_rng(self.seed)
        self.reset()
        
        
        self.size=(self.nos_states, self.nos_actions, self.nos_states)
        if not(use_integer_reward is None):
            self.R = (  integer_reward_matrix(self.size, self.reward_range) \
                        if use_integer_reward else \
                        real_reward_matrix(self.size, self.reward_range)
                    )
        else:
            self.R = None
    
        self.tr_dist=tr_dist
        self.use_logits = use_logits
        if not (tr_dist is None):
            tr_name, tr_arg1, tr_arg2 = tr_dist
            transitional_generator = getattr(TransitionGenerator, tr_name)
            self.transitional = \
                [([ 
                    transitional_generator(self.nos_states, tr_arg1, tr_arg2) \
                        for _ in range(self.nos_actions)]) \
                            for _ in range(self.nos_states)]
            self.Tr = self.generateTR()
        else:
            self.transitional = None
            self.Tr = None

    def generateTR(self):
        return [([ 
                (td.Categorical(logits = self.transitional[s][a]) \
                    if self.use_logits else \
                        td.Categorical(probs = self.transitional[s][a])) \
                    for a in range(self.nos_actions)]) for s in range(self.nos_states)]


    def clone(self):
        res = randMDP(self.nos_states, self.nos_actions, self.initial_states, self.final_states,
        max_ts=self._max_episode_steps, reward_range= self.reward_range, use_integer_reward=None, 
        tr_dist=None, use_logits=self.use_logits, seed=self.seed)
        res.R = np.copy(self.R)
        res.transitional = [([ 
                    self.transitional[s][a].clone() \
                        for a in range(self.nos_actions)]) \
                            for s in range(self.nos_states)]
        res.Tr = res.generateTR()
        return res

    def show_tr(self, p=print):
        for s in range(self.nos_states):
            for a in range(self.nos_actions):
                p(f'{s=}, {a=}, probs={self.Tr[s][a].probs}')

    def reset(self):
        self.S*=0
        self.S+=self.rng.choice(self.initial_states)
        self.ts = 0 # timestep
        self.x, self.y = [self.S[0]], [self.ts]
        self.ahist = [-1]
        return self.S 
        
    def step(self, action):
        # first check current level
        S = int(self.S)
        nS = int(self.Tr[S][action].sample().item())
        reward = self.R[S, action, nS]
        self.ts+=1
        # check if ns in in self.final states
        done = ((self.ts>=self._max_episode_steps) or (nS in self.final_states))
        self.S[:]=nS 
        self.x.append(nS)
        self.y.append(self.ts)
        self.ahist.append(action)
        return self.S, reward, done, {}

    def render(self):
        #s = 0
        fig = plt.figure(figsize=(6,6))
        plt.ylim(-1, self.nos_states)
        plt.scatter(self.y, self.x, marker='s')
        plt.plot(self.y, self.x)

        lwid = len(self.ahist)
        for n in self.initial_states:
            plt.hlines(n, 0,lwid, linestyles='dotted')
        for n in self.final_states:
            plt.hlines(n, 0,lwid, linestyles='solid')

        for n in range(1, len(self.ahist)):
            plt.annotate(str(self.ahist[n]), (self.y[n]-0.5, -0.5))

        plt.show()


    def fully_deterministic(nos_states, nos_actions, initial_states, final_states, max_ts,
                            reward_range, use_integer_reward, seed=None):
        return randMDP( nos_states, 
                        nos_actions, 
                        initial_states,
                        final_states,
                        max_ts,
                        reward_range,
                        use_integer_reward,
                        tr_dist=('deterministic', 0.0, 1.0),
                        use_logits=False, seed=seed )

    def equal_probability(nos_states, nos_actions, initial_states, final_states, max_ts,
                            reward_range, use_integer_reward, seed=None):
        return randMDP( nos_states, 
                        nos_actions, 
                        initial_states,
                        final_states,
                        max_ts,
                        reward_range,
                        use_integer_reward,
                        tr_dist=('similar', 0.0, 1.0),
                        use_logits=True, seed=seed )
#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE:

"""
#-----------------------------------------------------------------------------------------------------
