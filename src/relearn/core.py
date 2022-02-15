#-----------------------------------------------------------------------------------------------------
# relearn/core.py
#-----------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------
import torch
#-----------------------------------------------------------------------------------------------------

class OBJECT:
    """ 
    [OBJECT] - an empty object, can be used for a variety of purposes 
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
    [SPACE] - represents a vector space 

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
        (3) zeron(): creates and returns n zero-tensors     :: space.zeron(device='cpu'): -> torch.Tensor
    """
    NEW = lambda T: SPACE(T[0], T[1], T[2], T[3], T[4])
    def __init__(self, shape, dtype, low=0, high=0, discrete=False):
        self.shape , self.dtype, self.low, self.high, self.discrete, self.scalar = shape, dtype, low, high, discrete, (len(shape)==0) 
    def zero(self, device='cpu'):
        return torch.zeros(size=self.shape, dtype=self.dtype, device=device)
    def zeros(self, device='cpu'):
        return  torch.zeros(size=self.shape, dtype=self.dtype, device=device),\
                torch.zeros(size=self.shape, dtype=self.dtype, device=device)
    def zeron(self, n, device='cpu'):
        return torch.zeros(size=(n,)+self.shape, dtype=self.dtype, device=device)
    def __str__(self):
        return '[SPACE] : shape:[{}], dtype:[{}], discrete[{}], scalar:[{}]'.format(
                self.shape, self.dtype, self.discrete, self.scalar)
    def __repr__(self):
        return self.__str__()

class BUFFER:
    """
    [BUFFER] - a pair of tensors with attached memory, used by SIMULATOR class

    * BUFFER object has following attributes:
        [1] space:      underlying SPACE object
        [2] data:       current tensor
        [3] idata:      initial tensor
        [4] mem:        memory tensor

    * SPACE object has following functions:
        (1) copi():-> None      - copies initial tensor (idata) to current tensor (data)
        (2) snap(i):-> None     - copies current tensor (data) to memory tensor (mem) at index i
    """
    def __init__(self, space, double, snapable, capacity, buffer_device, memory_device):
        """
        Args
            space:      [SPACE]     underlying SPACE object
            double:     [bool]      if True, maintains a initial tensor (idata), uses self.copi to copy data 
            snapable:   [bool]      if True, snaps this tensor to memory when self.snap is called
            capacity:   [int]       the size of memory tensor

        * buffer_device, memory_device are torch devices. Note that these can be different, usually buffer should be kept on cpu
        """
        self.space = space
        self.data = space.zero(buffer_device)
        self.idata = space.zero(buffer_device) if double else None
        self.mem = space.zeron(capacity, memory_device) if snapable else None
    def copi(self):
        self.data.copy_(self.idata)
    def snap(self, i):
        self.mem[i].copy_(self.data)

class SIMULATOR:
    """
    [SIMULATOR] - collection of BUFFER objects to simulate an environment, each ENV object has its own SIMULATOR object

    * SIMULATOR object has following attributes:
        [1] buffer_device:      [str]   torch device for buffer tensors
        [2] memory_device:      [str]   torch device for memory tensors
        [4] capacity:           [int]   first dimension (count) of memory tensors
        [4] at_max:             [bool]  a flag indicating in memory is full
        [4] ptr:                [int]   pointer to end of memory
        [3] keys_all:           [set]   set of all buffer keys
        [4] keys_double:        [set]   set of buffers that have idata (copi() on call)
        [4] keys_snapable:      [set]   set of all buffer keys that have mem (snap() on call)

    * SIMULATOR class should not be created directly, as it requires initialization and then adding of buffers.
        Simulator is automatically created in the ENV class. Methods provided in ENV class can be used in all inherited instances
    """
    def __init__(self, buffer_device, memory_device, capacity):
        self.buffer_device, self.memory_device, self.capacity  = buffer_device, memory_device, capacity
        self.keys_all, self.keys_double, self.keys_snapable = set(), set(), set()
        self.clear_memory()

    def clear_memory(self): 
        """ clears memory of all buffers - actually just resets the pointers """
        self.at_max, self.ptr = False, 0

    def count(self):
        return self.capacity if self.at_max else self.ptr

    def add_buffer(self, key, space, double, snapable):
        if hasattr(self, key):
            raise NameError('SIMULATOR already has an attribute [{}], choose a different buffer key.'.format(key))
        else:
            setattr(self, key, BUFFER(space, double, snapable, self.capacity, self.buffer_device, self.memory_device))
            self.keys_all.add(key)
            self.keys_double.add(key) if double else None
            self.keys_snapable.add(key) if snapable else None

    def remove_buffer(self, key):
        if not hasattr(self, key):
            raise NameError('SIMULATOR does not have an attribute [{}], could not remove.'.format(key))
        else:
            self.keys_all.remove(key) if key in self.keys_all else None
            self.keys_double.remove(key) if key in self.keys_double else None
            self.keys_snapable.remove(key) if key in self.keys_snapable else None
            delattr(self, key)

    def copi(self):
        for key in self.keys_double:
            getattr(self, key).copi()

    def snap(self):
        for key in self.keys_snapable:
            getattr(self, key).snap(self.ptr)
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0

    def range_recent(self, size): # returns low, high, count
        return  (self.ptr - min( self.capacity, size ) + 2, self.ptr, self.capacity) if self.at_max else (self.ptr - min( self.ptr,      size ) + 1, self.ptr, self.ptr) 

    def range_random(self): # returns low, high, count
        return  ( self.ptr - self.capacity + 2, self.ptr, self.capacity ) if self.at_max else ( 1, self.ptr, self.ptr ) 

    def read(self, indices):
        return { key : getattr(self, key).mem[indices] for key in self.keys_snapable }

    def read_(self, indices):
        return OBJECT( **self.read(indices) )

    def render_buffer(self, p=print):
        p('=-=-=-=-==-=-=-=-=@BUFFER=-=-=-=-==-=-=-=-=')
        p("Keys-[{}: {}]\nDouble-[{}: {}]\nSnapable-[{}: {}]".format(
            len(self.keys_all), self.keys_all, len(self.keys_double), self.keys_double, len(self.keys_double), self.keys_double  ) )
        p('------------------@KEYS------------------')
        for key in self.keys_all:
            p('\t{}\t::\t{}'.format(key, getattr(self, key).val) )
        p('=-=-=-=-==-=-=-=-=!KEYS=-=-=-=-==-=-=-=-=')

    def render_memory(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@MEMORY=-=-=-=-==-=-=-=-=')
        p("Count ["+str(self.count())+"]\nCapacity ["+str(self.capacity)+ "]\nPointer ["+str(self.ptr)+ "]")
        p('------------------@SLOTS------------------')
        for i in range (low, high, step):
            p('[SLOT: {}]\n'.format(i))
            for key in self.keys_snapable:
                p('\t{}\t::\t{}'.format(key, getattr(self, key).mem[i]))
        p('=-=-=-=-==-=-=-=-=!MEMORY=-=-=-=-==-=-=-=-=')

    def render_memory_all(self, p=print):
        self.render_memory(0, self.count(), p=p)

    def render_memory_last(self, nos, p=print):
        self.render_memory(-1, -nos-1, step=-1,  p=p)

class ENV:
    """
    [ENV] - simulates an environment using a SIMULATOR object. Each ENV objects has its own SIMULATOR object

    * ENV object has following attributes:
        [1] known:          [Any]           an object that contains 'knowledge' common for all ENV instances
        [2] task:           [Any]           an object that contains 'task' that is ENV instance-specific
        [3] sim:            [SIMULATOR]     underlying SIMULATOR object
        [4, 5] SKey, S:     [str, BUFFER]   a buffer representing the observation (visible to agent)
        [6, 7] AKey, A:     [str, BUFFER]   a buffer representing the action (taken by the agent)
        [8, 9] FKey, F:     [str, BUFFER]   a buffer representing flag (type of state) 
        [10] auto_snap:     [bool]          if True, stores buffer snaps into memory (only those which are 'snapable') on every reset() and step()
        [11] flag:          [int]           represets type of state {-1: 'initial', 0: 'non-terminal', 1:'terminal'}
        [12] steps:         [int]           timesteps elapsed since last call to restart()
    
    * Inherit the ENV class to implement custom environments. 
    * Inherited class must implement the functions marked as NOT IMPLEMENTED (ending with a capital 'F') 
    """
    def __init__(self,  known, task, 
                        buffers, state_key, action_key, flag_key, buffer_device, 
                        memory_device, capacity, auto_start=True, auto_snap=True ):
        self.known, self.task =  known, task
        self.initF()
        
        self.sim = SIMULATOR(buffer_device, memory_device, capacity)
        for key, space, double, snapable in buffers:
            self.sim.add_buffer(key, space, double, snapable)

        self.Skey, self.Akey, self.Fkey = state_key, action_key, flag_key
        self.S = getattr(self.sim, self.Skey)
        self.A = getattr(self.sim, self.Akey)
        self.F = getattr(self.sim, self.Fkey)
    
        self.auto_snap=auto_snap
        self.flag = None
        self.steps = -1
        self.reset() if auto_start else None

    def reset(self):
        self.resetF()
        self.restart()

    def restart(self):
        self.sim.copi()

        self.flag = self.restartF()
        self.steps = 0

        self.F.val.fill_(self.flag)
        self.sim.snap() if self.auto_snap else None

    def step(self, action):
        self.A.val.copy_(action)

        self.flag = self.stepF()
        self.steps+=1

        self.F.val.fill_(self.flag)
        self.sim.snap() if self.auto_snap else None
    
    def explore_steps(self, policy, moves, max_steps, frozen):
        for _ in range(moves):
            if (self.flag or self.steps >= max_steps):
                self.restart() if frozen else self.reset()
            self.step(policy.predict(self.S.val))
        return

    def explore_episodes(self, policy, moves, max_steps, frozen):
        (self.restart() if frozen else self.reset()) if (self.flag or self.steps >= max_steps) else (None)    
        for _ in range(moves):
            while not (self.flag or self.steps >= max_steps):
                self.step(policy.predict(self.S.val))
            self.restart() if frozen else self.reset()
        return


    def initF(self):
        print(' !-- WARNING: NOT IMPLEMENTED --! [initF] ')
        return
    def resetF(self):
        print(' !-- WARNING: NOT IMPLEMENTED --! [resetF] ')
        return
    def restartF(self):
        flag = None
        print(' !-- WARNING: NOT IMPLEMENTED --! [restartF] ')
        return flag
    def stepF(self):
        flag = None
        print(' !-- WARNING: NOT IMPLEMENTED --! [stepF] ')
        return flag

class PIE:
    """
    [PIE] - represent an abstract policy

    * PIE object has following attributes:
        [1] device:          [str]           torch.device
        [2] state_space:     [SPACE]         underlying state space
        [3] action_space:    [SPACE]         underlying action space
    
    * Inherit the PIE class to implement custom policies. 
    * Inherited class must implement the functions marked as NOT IMPLEMENTED (ending with a capital 'F') 
    """
    def __init__(self, device, state_space, action_space):
        self.device, self.state_space, self.action_space = device, state_space, action_space
        self.action = self.action_space.zero()

    def predict(self, state): 
        # state is an buffer tensor usually, env.buffer.S
        self.predictF(state)
        return self.action

    def predictF(self, state):
        self.action*=0
        print(' !-- WARNING: NOT IMPLEMENTED --! [stepF] ')
        return
            

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


