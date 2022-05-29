
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# exp.py :: explorers and memory for training and testing on gym-like environments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch as tt
import numpy as np
from .common import observation_key, action_key, reward_key, done_key, step_key
from .common import default_spaces, space_range, REX



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A] Static Replay Memory """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Ω:
    """ [Ω] - A key based static replay memory """

    def __init__(self, capacity, named_spaces, seed) -> None:
        """ named_spaces is a dict like string_name vs gym.space """
        assert(capacity>0)
        self.capacity, self.spaces = capacity+2, named_spaces
        # why add2 to capacity -> one transition requires 2 slots and we need to put an extra slot for current pointer
        self.rng = np.random.default_rng(seed)
        self.build_memory()
        
    def build_memory(self):
        self.data={}
        for key, space in self.spaces.items():    
            if key!='':
                self.data[key] = np.zeros((self.capacity,) + space.shape, space.dtype)
        self.ranger = np.arange(0, self.capacity, 1)
        self.mask = np.zeros(self.capacity, dtype=np.bool8) #<--- should not be changed yet
        self.keys = self.data.keys()
        self.clear()

    def clear(self):
        self.at_max, self.ptr = False, 0
        self.mask*=False

    def length(self): # Excludes the initial observation (true mask only)
        return len(self.ranger[self.mask])

    def count(self): # Includes the initial observation (any mask)
        return self.capacity if self.at_max else self.ptr
    
    def snap(self, mask, **info):
        """ snaps all keys in self.keys - assuming info has the keys already """
        for k in self.keys:
            self.data[k][self.ptr] = info[k]
        self.mask[self.ptr] = mask
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return

    """ NOTE: Sampling

        > sample_methods will only return indices, use self.read(i) to read actual tensors
        > Valid Indices - indices which can be choosen from, indicates which transitions should be considered for sampling
            valid_indices = lambda : self.ranger[self.mask]
    """    

    def sample_recent(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = np.where(valid_indices==self.ptr)[0] # find index of self.ptr in si
        pick = min ( len(valid_indices)-1, size )
        return valid_indices[ np.arange(iptr-pick, iptr, 1) ]

    def sample_recent_(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = np.where(valid_indices==self.ptr)[0] # find index of self.ptr in si
        pick = min ( len(valid_indices)-1, size )
        return pick, valid_indices[ np.arange(iptr-pick, iptr, 1) ]

    def sample_all_(self):
        return self.sample_recent_(self.length())

    def sample_all(self):
        return self.sample_recent(self.length())

    def sample_random(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min ( len(valid_indices), size )
        return self.rng.choice(valid_indices, pick, replace=replace)

    def sample_random_(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min ( len(valid_indices), size )
        return pick, self.rng.choice(valid_indices, pick, replace=replace)

    def read(self, i): # reads [all keys] at [given sample] indices
        return { key:self.data[key][i] for key in self.keys }

    def readkeys(self, i, keys): # reads [given keys] at [given sample] indices
        return { key:self.data[key][i] for key in keys }

    def readkeis(self, ii, keys, teys): # reads [given keys] at [given sample] indices and rename as [given teys]
        return { t:self.data[k][i] for i,k,t in zip(ii,keys,teys) }

    def readkeist(self, *args): # same as 'readkeis' but the args are tuples like: (index, key, tey)
        return { t:self.data[k][i] for i,k,t in args }
                

        
    """ NOTE: Rendering """
    def render(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@[MEMORY]=-=-=-=-==-=-=-=-=')
        p("Length:[{}]\tCount:[{}]\nCapacity:[{}]\tPointer:[{}]".format(self.length(), self.count(), self.capacity, self.ptr))
        for i in range (low, high, step):
            p('____________________________________')  #p_arrow=('<--------[PTR]' if i==self.ptr else "")
            if self.mask[i]:
                p('SLOT: [{}]+'.format(i))
            else:
                p('SLOT: [{}]-'.format(i))
            for key in self.data:
                p('\t{}: {}'.format(key, self.data[key][i]))
        p('=-=-=-=-==-=-=-=-=![MEMORY]=-=-=-=-==-=-=-=-=')

    def render_all(self, p=print):
        self.render(0, self.count(), p=p)

    def render_last(self, nos, p=print):
        self.render(-1, -nos-1, step=-1,  p=p)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B] Base Explorer Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ξ: 
    """ [ξ] - An explorer with memory that can explore an environment using policies 
              - remember to set self.pie before exploring
              - setting any policy to None will cause it default to self.random()
    """

    def __init__(self, env, episodic, memory_capacity, memory_seed, snap_enabled, **extra_spaces) -> None:
        
        self.spaces={}
        self.spaces.update(default_spaces(env.observation_space, env.action_space))
        self.spaces.update(extra_spaces)
        
        self.alow, self.ahigh = space_range( self.spaces[action_key] )
        self.olow, self.ohigh = space_range( self.spaces[observation_key] )
        
        self.false_extra_snap = {k: np.zeros(shape=space.shape,dtype=space.dtype) for k,space in extra_spaces.items()} 
        self.false_act_snap = np.zeros(shape=self.spaces[action_key].shape, dtype=self.spaces[action_key].dtype)
        self.false_reward_snap = np.zeros(shape=self.spaces[reward_key].shape, dtype=self.spaces[reward_key].dtype)

        self.env = env
        self.episodic = episodic
        self.explore = self.explore_episodes if episodic else self.explore_steps

        self.memory = (Ω(memory_capacity, self.spaces, memory_seed ) if memory_capacity>0 else None)
        self.enable_snap() if snap_enabled else self.disable_snap()
        self.reset()
        
    def enable_snap(self):
        if not (self.memory is None):
            self.snap = self.do_snap
        else:
            self.disable_snap()

    def disable_snap(self):
        self.snap = self.no_snap

    def clear_snap(self):
        if not (self.memory is None):
            self.memory.clear()

    def reset(self, clear_memory=False):
        # Note: this does not reset env, just resets its buffer and flag
        self.cs, self.done, self.ts = None, True, 0
        self.act, self.reward = self.false_act_snap, self.false_reward_snap
        self.clear_snap() if clear_memory else None
        self.N = 0 #<--- is update after completion of episode or step based on (self.episodic)

    def no_snap(self, mask):
        return

    def do_snap(self, mask):
        self.memory.snap(mask, **{   
                observation_key : self.cs,
                action_key :      self.act ,
                reward_key :      self.reward ,
                done_key :        self.done,
                step_key:         self.ts,
                **self.info
                })

    @tt.no_grad()
    def explore_steps(self, N):
        for _ in range(N):
            if self.done:
                self.cs, self.act, self.reward, self.done, self.ts, self.info = self.env.reset(), 0, 0, False, 0, self.false_extra_snap
                self.snap(False)

            self.act = self.get_action()
            self.cs, self.reward, self.done, self.info = self.env.step(self.act)
            self.ts+=1
            self.snap(True)
            self.N+=1
        return N

    @tt.no_grad()
    def explore_episodes(self, N):
        n = self.ts
        for _ in range(N):
            self.cs, self.act, self.reward, self.done, self.ts, self.info = self.env.reset(), 0, 0, False, 0, self.false_extra_snap
            self.snap(False)

            while not self.done:
                self.act = self.get_action()
                self.cs, self.reward, self.done, self.info = self.env.step(self.act)
                self.ts+=1
                self.snap(True)
            self.N+=1
        return self.ts - n

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [C] Derived Explorer Classes """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class randomX(ξ):
    # Random explorer - explorers based on inbuild env.action_space.sample

    def set_policy(self, pie):
        print(f'Warning: setting policy [{pie}] on a random explorer, this has no effect.')
        pass

    def get_action(self): 
        return self.env.action_space.sample()

class policyX(ξ): 
    # Policy explorer - explorers based on a given policy

    def set_policy(self, pie):
        if not hasattr(pie, 'predict'):
            raise REX(f'policy object [{pie}] does not implement predict.')
        self.pie = pie

    def get_action(self):
        return self.pie.predict(self.cs)

class greedyX(ξ): 
    # Greedy-Epsilon explorer - explorers based on 2 policies and explorarion probability (epsilon)
    # epsilonF(self.N) -> bool :: asks if to explore or not
    def set_policy(self, pie, epsilonF):
        if not hasattr(pie, 'predict'):
            raise REX(f'policy object [{pie}] does not implement predict.')
        self.pie = pie
        self.do_explore = lambda : epsilonF(self.N)

    def get_action(self):
        return ( self.env.action_space.sample() if self.do_explore() else self.pie.predict(self.cs) )

class noisyX(ξ): 
    # Noisy explorer - explorers based on a single policy with added noise
    # noiseF(self.N) -> vector :: return a noise that is directly added to action
    def set_policy(self, pie, noiseF):
        if not hasattr(pie, 'predict'):
            raise REX(f'policy object [{pie}] does not implement predict.')
        self.pie = pie
        self.action_noise = lambda : noiseF(self.N)

    def get_action(self):
        return ( self.pie.predict(self.cs) + self.action_noise() )


#-----------------------------------------------------------------------------------------------------
""" FOOT NOTE:

[ARCHIVE]
    def prepare_batch(self, batch_size, dtype, device, recent=False, 
                input_shape_observation=None, input_shape_action=None):

        pick, samples = (self.sample_recent_(size = batch_size)) \
                            if recent else \
                        (self.sample_random_(size=batch_size, replace=False))
        #if pick != batch_size:
        #    print('[!~WARNING] batch_size=[{}], pick_size=[{}]'.format(batch_size, pick))
        batch = self.readkeis(
            (samples-1,                  samples,                  samples,                 samples,            samples      ), 
            (observation_key,            observation_key,          action_key,              reward_key,         done_key     ), 
            ('cS',                       'nS',                     'A',                     'R',                'D'          ))

        cS = tt.tensor(batch['cS'], dtype=dtype, device=device)
        nS = tt.tensor(batch['nS'], dtype=dtype, device=device)
        if not (input_shape_observation is None):
            cS = tt.reshape(cS, (batch_size,) + input_shape_observation)
            nS = tt.reshape(nS, (batch_size,) + input_shape_observation)

        A = tt.tensor(batch['A'], dtype=tt.long, device=device)
        if not (input_shape_action is None):
            A = tt.reshape(A, (batch_size,) + input_shape_action)

        R = tt.tensor(batch['R'], dtype=dtype, device=device)
        D = tt.tensor(batch['D'], dtype=dtype, device=device)
        
        return pick, cS, nS, A, R, D

"""
#-----------------------------------------------------------------------------------------------------