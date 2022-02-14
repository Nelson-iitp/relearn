#-----------------------------------------------------------------------------------------------------
# relearn/models.py
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
# IMPORTS
import torch
from .core import SPACE

#-----------------------------------------------------------------------------------------------------
def strScalar(t):
    return str(t.item())
def strVector(t, sep=","):
    res=""
    for ti in t.flatten():
        res+=str(ti.item())+sep
    return res

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def mul_(A):
    res = 1
    for a in A.flatten():
        res*=a.item()
    return res
def int2baseA_(num, baseX, device, reshaper):
    base = baseX.flatten()
    digs = len(base)
    res = base.clone().to(device)
    q = num
    for i in range(digs):
        res[i]=q%base[i].item()
        q = torch.floor(q/base[i]).item()
    return res.reshape(reshaper)
def ENUM_SPACE(space, device='cpu'):
    assert(space.discrete) #<-- this should be true
    low, high = space.zeros(device) 
    low += torch.tensor(space.low, dtype=space.dtype, device=device)
    high += torch.tensor(space.high, dtype=space.dtype, device=device)
    delta = high-low #<--- make sure its all positive
    elements = []
    if space.scalar:
        all = int(delta.item())
        for a in range(all):
            elements.append( torch.tensor(a , device=device) + low )
    else:
        all = int(mul_(delta))
        for a in range(all):
            elements.append( int2baseA_(a,delta, device, space.shape)+low )
    return len(elements), low, high, delta, elements
    
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class PFUNC():
    """ state-action function in a table """
    def __init__(self, state_space, action_space, device, do_enum=False):

        self.device = device
        self.action_space = action_space

        self.out_space = action_space
        self.out_zero = self.out_space.zero(self.device)

        self.state_space = state_space
        self.mapperS = strScalar if self.state_space.scalar else strVector
        self.F = {} # Value dictionary

        if do_enum:
            assert (state_space.discrete) #<---- must
            self.nS , self.lS, self.hS, self.dS, self.S = ENUM_SPACE(self.state_space, self.device) # discrete -> shape
            for s in self.S:
                self.F[self.mapperS(s)] = self.out_space.zero(self.device)


    def call(self, state):
        cS = self.mapperS(state)
        return self.F[cS] if cS in self.F else self.out_zero



class VFUNC():
    """ state-value function in a table """
    def __init__(self, state_space, device, dtype=torch.float32, do_enum=False):
        self.device = device
        self.out_space = SPACE.NEW(((), dtype, 0, 0, False))
        self.out_zero = self.out_space.zero(self.device)

        self.state_space = state_space
        self.mapperS = strScalar if self.state_space.scalar else strVector
        self.F = {} # Value dictionary

        if do_enum:
            assert (state_space.discrete) #<---- must
            self.nS, self.lS, self.hS, self.dS, self.S = ENUM_SPACE(self.state_space, self.device) # discrete -> shape
            for s in self.S:
                self.F[self.mapperS(s)] = self.out_space.zero(self.device)
            # del self.S #<--- do not delete, may be useful

    def call(self, state):
        cS = self.mapperS(state)
        return self.F[cS] if cS in self.F else self.out_zero

class QFUNC():
    """ state-action-value function in a table """
    def __init__(self, state_space, action_space, device, dtype=torch.float32, default_call=False, do_enum=False):
        assert (action_space.discrete) #<---- must
        self.device = device
        self.action_space = action_space
        self.mapperA = strScalar if self.action_space.scalar else strVector
        self.nA, self.lA, self.hA, self.dA, self.A = ENUM_SPACE(self.action_space, self.device) # discrete -> shape
        self.AD = {}
        for i,a in enumerate(self.A):
            self.AD[self.mapperA(a)] = i
        # del self.A #<-- do not delete, usefull in self.call
        self.out_space = SPACE.NEW(((self.nA,), dtype, 0, 0, False))
        self.out_zero = self.out_space.zero(self.device)

        self.state_space = state_space
        self.mapperS = strScalar if self.state_space.scalar else strVector
        self.F = {} # Value dictionary

        if do_enum:
            assert (state_space.discrete) #<---- must
            self.nS , self.lS, self.hS, self.dS, self.S = ENUM_SPACE(self.state_space, self.device) # discrete -> shape
            for s in self.S:
                self.F[self.mapperS(s)] = self.out_space.zero(self.device)
        self.default_call = default_call
        self.call = self.callSA if self.default_call else self.callS

    def callSA(self, state, action):
        cS = self.mapperS(state)
        cA = self.AD[self.mapperA(action)]
        return self.F[cS][cA] if cS in self.F else self.out_zero[cA]

    def callS(self, state):
        cS = self.mapperS(state)
        return self.F[cS] if cS in self.F else self.out_zero

#-----------------------------------------------------------------------------------------------------


