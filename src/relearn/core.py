#-----------------------------------------------------------------------------------------------------
# relearn/core.py
#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
import torch
from numpy.random import default_rng
#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
# Basic Classes
#-----------------------------------------------------------------------------------------------------

class OBJ:
    """ 
    [OBJ] - an empty object, can be used for a variety of purposes 
    """
    def __init__(self, **kwarg):
        for arg in kwarg:
            setattr(self, arg, kwarg[arg])
        pass #print('[{}]\t ({}) => ({})'.format(i, arg, kwarg[arg]))
    def __str__(self) -> str:
        res="=-=-=-=-==-=-=-=-=\n__DICT__: "+str(len(self.__dict__))+"\n=-=-=-=-==-=-=-=-=\n"
        for i in self.__dict__:
            res+=str(i) + '\t:\t' + str(self.__dict__[i]) + "\n"
        return res + "=-=-=-=-==-=-=-=-=\n"
    def __repr__(self) -> str:
        return self.__str__()

class SPACE:
    """ 
    [SPACE] - equivalent of a vector space 

    * SPACE object has following attributes:
        [1] shape:      dimension (tuple)
        [2] dtype:      data-type (torch.dtype)
        [3] low:        lower bound - can be a scalar or an array (broadcast rules apply)
        [4] high:       upper bound - can be a scalar or an array (broadcast rules apply)
        [5] discrete    if True, indicates that all dimensions take discrete integer values
        [6] scalar      (len(shape)==0)

    * SPACE object has following functions:
        (1) zero(): creates and returns one zero-tensor     :: space.zero(device='cpu'):  -> torch.Tensor
        (2) zeros(): creates and returns two zero-tensors   :: space.zeros(device='cpu'): -> torch.Tensor, torch.Tensor
    """
    def __init__(self, shape, dtype, low=0, high=0, discrete=False):
        self.shape , self.dtype = shape, dtype                   # primary
        self.low, self.high, self.discrete = low, high, discrete # user
        self.scalar = (len(self.shape)==0)                       # derived
    def zero(self, device='cpu'):
        return torch.zeros(size=self.shape, dtype=self.dtype, device=device)
    def zeros(self, device='cpu'):
        return  torch.zeros(size=self.shape, dtype=self.dtype, device=device),\
                torch.zeros(size=self.shape, dtype=self.dtype, device=device)
    def __str__(self):
        return '[SPACE] : shape:[{}], dtype:[{}], discrete[{}], scalar:[{}]'.format(
                self.shape, self.dtype, self.discrete, self.scalar)
    def __repr__(self):
        return self.__str__()
    def NEW(info):
        """ takes 5-tuple: info = ( shape, dtype, low, high, discrete) and creates new SPACE object """
        return SPACE(info[0], info[1], info[2], info[3], info[4])
    def SPACES( hidden_info =   ((), torch.float32, 0, 0, False), 
                state_info =    ((), torch.float32, 0, 0, False), 
                action_info =   ((), torch.int32,   0, 2, True),
                reward_info =   ((), torch.float32, 0, 0, False),
                done_info =     ((), torch.bool,    0, 0, True),
                counter_info =  ((), torch.int32,   0, 0, True),
                tag_info =      ((), torch.bool,    0, 0, True)
                ):
        """
        * returns a standard set of nessesary 'spaces' (returns OBJ)
        * 'spaces' object has 7 standard spaces: ( H, S, A, R, D, T, G ) used to create BUFFER objects in ENV
            [H]idden    :   part of state that is hidden from agent     (hidden shape and dtype can be choosen)
            [S]tate     :   part of state that is visible to agent      (state shape and dtype can be choosen)
            [A]ction    :   action-vector that the agent can write to   (action shape and dtype can be choosen)
            [R]eward    :   reward-vector that agent can read from      (reward shape is (), can only choose dtype)
            [D]one      :   done flag indicating final state            (done shape is (), dtype is torch.bool)
            Coun[T]er   :   used to count timesteps                     (counter shape is (), can only choose dtype)
            Ta[G]       :   used store meta-data in memory              (tag shape and dtype can be choosen)

        """
        x = OBJ() # ( H, S, A, R, D, T, G )
        x.H, x.S, x.A, x.R, x.D, x.T, x.G  = SPACE.NEW(hidden_info), SPACE.NEW(state_info), SPACE.NEW(action_info), SPACE.NEW(reward_info), SPACE.NEW(done_info), SPACE.NEW(counter_info), SPACE.NEW(tag_info)
        return x


#-----------------------------------------------------------------------------------------------------
# Core Classes
#-----------------------------------------------------------------------------------------------------

class BUFFER:

    def __init__(self, spaces, capacity, device, memory_device, auto_create=True):
        self.device, self.memory_device = device, memory_device
        self.spaces = spaces
        self.capacity = capacity
        self.create_buffer()
        self.has_memory = False
        self.create_memory() if auto_create else None

    def create_buffer(self):
        # create buffer
        self.Hi, self.H = self.spaces.H.zeros(self.device)
        self.Si, self.S = self.spaces.S.zeros(self.device)
        self.Ai, self.A = self.spaces.A.zeros(self.device)
        self.Ri, self.R = self.spaces.R.zeros(self.device)
        self.Di, self.D = self.spaces.D.zeros(self.device)
        self.Ti, self.T = self.spaces.T.zeros(self.device)
        self.G = self.spaces.G.zero(self.device)

    def create_memory(self):
        # create memory
        self.SS = torch.zeros( size=(self.capacity,)+self.spaces.S.shape, dtype=self.spaces.S.dtype, device=self.memory_device )
        self.AA = torch.zeros( size=(self.capacity,)+self.spaces.A.shape, dtype=self.spaces.A.dtype, device=self.memory_device )
        self.RR = torch.zeros( size=(self.capacity,)+self.spaces.R.shape, dtype=self.spaces.R.dtype, device=self.memory_device )
        self.DD = torch.zeros( size=(self.capacity,)+self.spaces.D.shape, dtype=self.spaces.D.dtype, device=self.memory_device )
        self.TT = torch.zeros( size=(self.capacity,)+self.spaces.T.shape, dtype=self.spaces.T.dtype, device=self.memory_device )
        self.GG = torch.zeros( size=(self.capacity,)+self.spaces.G.shape, dtype=self.spaces.G.dtype, device=self.memory_device )
        self.clear_memory()
        self.has_memory = True

    def clear_memory(self):
        self.at_max = False
        self.ptr = 0

    def copi(self):
        self.H.data.copy_(self.Hi)
        self.S.data.copy_(self.Si)
        self.A.data.copy_(self.Ai)
        self.R.data.copy_(self.Ri)
        self.D.data.copy_(self.Di)
        self.T.data.copy_(self.Ti)

    def count(self):
        return self.capacity if self.at_max else self.ptr

    def snap(self): # begin a new item in list (new episode)
        self.SS[self.ptr].data.copy_(self.S)
        self.AA[self.ptr].data.copy_(self.A)
        self.RR[self.ptr].data.copy_(self.R)
        self.DD[self.ptr].data.copy_(self.D)
        self.TT[self.ptr].data.copy_(self.T)
        self.GG[self.ptr].data.copy_(self.G)
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max=True
            self.ptr = 0

    def SAMPLE_BATCH(self, rng, size):
        count = self.capacity if self.at_max else self.ptr
        return torch.tensor(rng.integers(self.ptr - count + self.at_max + 1, self.ptr, size=min( count, size )), device=self.memory_device)
    
    def SAMPLE_RECENT(self, size):
        count = self.capacity if self.at_max else self.ptr
        return torch.arange(self.ptr - min( count, size ) + self.at_max + 1, self.ptr, 1, device=self.memory_device)
    
    def PREPARE_BATCH(self, samples): # returns cS, A, R, D, nS
        actual_samples = []
        for i in samples:
            if self.TT[i]>0:
                actual_samples.append(i)
        si = torch.tensor(actual_samples, samples.dtype, device=samples.device)
        return self.SS[si-1], self.AA[si], self.RR[si], self.DD[si], self.SS[si] 

    def render(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@MEMORY=-=-=-=-==-=-=-=-=')
        p("Count ["+str(self.count())+"]\nCapacity ["+str(self.capacity)+ "]\nPointer ["+str(self.ptr)+ "]")
        p('------------------@SLOTS------------------')
        for i in range (low, high, step):
            p('Transition: [{}] :: T:[{}], S:[{}], A:[{}], R:[{}], D:[{}], G:[{}]'.format(
                    i, self.TT[i], self.SS[i], self.AA[i], self.RR[i], self.DD[i], self.GG[i]))
            
        p('=-=-=-=-==-=-=-=-=!MEMORY=-=-=-=-==-=-=-=-=')

    def render_all(self, p=print):
        self.render(0, self.count(), p=p)

    def render_last(self, nos, p=print):
        self.render(-1, -nos-1, step=-1,  p=p)

class ENV:

    def __init__(self, known, task, spaces, capacity, device, memory_device, seed=None, auto_create=True, auto_start=True, auto_snap=True):
        self.known, self.task, self.spaces =  known, task, spaces
        self.device = device 
        self.rng =  default_rng(seed)
        self.initF()
        self.buffer=BUFFER(spaces, capacity, device, memory_device, auto_create) #<--- buffer variable is initialized after initF because initF does not reuqire it
        self.started = False
        self.auto_snap=auto_snap
        self.start() if auto_start else None

    def start(self):
        self.reset()
        self.restart()
        self.started = True

    def reset(self):
        self.resetF()

    def restart(self):
        self.buffer.copi()
        self.restartF()
        self.buffer.snap() if self.auto_snap else None

    def step(self, action):
        self.buffer.A.data.copy_( action )
        self.stepF()
        self.buffer.T += 1
        self.buffer.snap() if self.auto_snap else None
    
    def explore_steps(self, policy, moves, max_steps, frozen):
        for _ in range(moves):
            if (self.env.buffer.D or self.env.buffer.T >= max_steps):
                self.reset() if not frozen else None
                self.restart()
            self.step(policy.predict(self.env.buffer.S))
        return

    def explore_episodes(self, policy, moves, max_steps, frozen):
        if (self.env.buffer.D or self.env.buffer.T >= max_steps):
            self.reset() if not frozen else None
            self.restart()
        for _ in range(moves):
            while not (self.env.buffer.D or self.env.buffer.T >= max_steps):
                self.step(policy.predict(self.env.buffer.S))
            self.reset() if not frozen else None
            self.restart()
        return

    def buildF( **kwargs ):
        spaces, known = SPACE.SPACES(), OBJ(**kwargs)
        print(' !-- WARNING: NOT IMPLEMENTED --! [buildF] ')
        return spaces, known
    def strF( state ):
        print(' !-- WARNING: NOT IMPLEMENTED --! [strF] ')
        return ""
    def initF(self):
        print(' !-- WARNING: NOT IMPLEMENTED --! [initF] ')
    def resetF(self):
        print(' !-- WARNING: NOT IMPLEMENTED --! [resetF] ')
    def restartF(self):
        print(' !-- WARNING: NOT IMPLEMENTED --! [restartF] ')
    def stepF(self):
        print(' !-- WARNING: NOT IMPLEMENTED --! [stepF] ')
    
class PIE:

    def __init__(self, device, spaces, seed=None):
        self.device, self.spaces = device, spaces
        self.rng = default_rng(seed)

    def predict(self, state): 
        # state is an buffer tensor usually, env.buffer.S
        self.action = None
        self.predictF(state)
        return self.action

    def predictF(self, state):
        # by default implements uniform random policy
        if self.spaces.A.discrete:
            self.action = torch.tensor(self.rng.integers(self.spaces.A.low, self.spaces.A.high, size=self.spaces.A.shape),
                                dtype=self.spaces.A.dtype, device=self.device )
        else:
            self.action = torch.tensor(self.rng.uniform(self.spaces.A.low, self.spaces.A.high, size=self.spaces.A.shape),
                                dtype=self.spaces.A.dtype, device=self.device )
            


#-----------------------------------------------------------------------------------------------------
# Author: Nelson Sharma
#-----------------------------------------------------------------------------------------------------
# NOTE 
""" 
1. On use of numpy.random.default_rng()
    `` numpy has better reproducebility through RNGs while torch does not gurantee reproducebility ``
        --> according to offical website (https://pytorch.org/docs/stable/notes/randomness.html)

2. On Choosing torch Devices: 'exploration' of an environment might not benefit from GPU in most cases 
    * as it requires reading and writing often ( on every state-transition )
    * and it increases data movement between CPU and GPU
    * only replay buffer and/or the policy parmeters should reside on GPU
    * 'exploration' can be instead facilitated using cpu based multi-threading or multi-processing

 """
#-----------------------------------------------------------------------------------------------------


