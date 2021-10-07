import numpy as np
import chainer
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
#from chainer import optimizers
#import matplotlib.pyplot as plt

# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - 
# [*] Define DQN Structure
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - 

class Qnet_MLP_1L_RELU(Chain):
    "chainer implementation"
    def __init__(self, state_dim, L1, action_dim):
        super(Qnet_MLP_1L_RELU, self).__init__()
        with self.init_scope():
            self._1 = L.Linear(state_dim, L1)
            self._o = L.Linear(L1, action_dim)
    def forward(self, x):
        h = F.relu(self._1(x))
        return self._o(h)
    
    def get_weights(self):
        return [
            np.copy(self._1.W.array), np.copy(self._1.b.array),
            np.copy(self._o.W.array), np.copy(self._o.b.array),
          ]
    def compar_weights(self, wA):
        me = self.get_weights()
        diffs=0
        for i in range(len(wA)):
            diffs+=(np.sum(np.abs(me[i]-wA[i])))
        return diffs
    def info(self):        
        a=[self._1.W.array.shape, self._1.b.array.shape]
        d=[self._o.W.array.shape, self._o.b.array.shape]
        ash=(a[0][0]*a[0][1]) + (a[1][0])
        dsh=(d[0][0]*d[0][1]) + (d[1][0])
        nos_params = ash+dsh
        print('Layers-----------------')
        print(' Layer[1]\t',a,ash,'\n',
               'Layer[o]\t',d,dsh,'\n',
              'Total Params:',nos_params)
              
class Qnet_MLP_2L_RELU(Chain):
    "chainer implementation"
    def __init__(self, state_dim, L1, L2, action_dim):
        super(Qnet_MLP_2L_RELU, self).__init__()
        with self.init_scope():
            self._1 = L.Linear(state_dim, L1)
            self._2 = L.Linear(L1, L2)
            self._o = L.Linear(L2, action_dim)
    def forward(self, x):
        h = F.relu(self._1(x))
        h = F.relu(self._2(h))
        return self._o(h)
    
    def get_weights(self):
        return [
            np.copy(self._1.W.array), np.copy(self._1.b.array),
            np.copy(self._2.W.array), np.copy(self._2.b.array),
            np.copy(self._o.W.array), np.copy(self._o.b.array),
          ]
    def compar_weights(self, wA):
        me = self.get_weights()
        diffs=0
        for i in range(len(wA)):
            diffs+=(np.sum(np.abs(me[i]-wA[i])))
        return diffs
    def info(self):        
        a=[self._1.W.array.shape, self._1.b.array.shape]
        b=[self._2.W.array.shape, self._2.b.array.shape]
        d=[self._o.W.array.shape, self._o.b.array.shape]
        ash=(a[0][0]*a[0][1]) + (a[1][0])
        bsh=(b[0][0]*b[0][1]) + (b[1][0])
        dsh=(d[0][0]*d[0][1]) + (d[1][0])
        nos_params = ash+bsh+dsh
        print('Layers-----------------')
        print(' Layer[1]\t',a,ash,'\n',
               'Layer[2]\t',b,bsh,'\n',
               'Layer[o]\t',d,dsh,'\n',
              'Total Params:',nos_params)
              
class Qnet_MLP_3L_RELU(Chain):
    "chainer implementation"
    def __init__(self, state_dim, L1, L2, L3, action_dim):
        super(Qnet_MLP_3L_RELU, self).__init__()
        with self.init_scope():
            self._1 = L.Linear(state_dim, L1)
            self._2 = L.Linear(L1, L2)
            self._3 = L.Linear(L2, L3)
            self._o = L.Linear(L3, action_dim)
    def forward(self, x):
        h = F.relu(self._1(x))
        h = F.relu(self._2(h))
        h = F.relu(self._3(h))
        return self._o(h)
    
    def get_weights(self):
        return [
            np.copy(self._1.W.array), np.copy(self._1.b.array),
            np.copy(self._2.W.array), np.copy(self._2.b.array),
            np.copy(self._3.W.array), np.copy(self._3.b.array),
            np.copy(self._o.W.array), np.copy(self._o.b.array),
          ]
    def compar_weights(self, wA):
        me = self.get_weights()
        diffs=0
        for i in range(len(wA)):
            diffs+=(np.sum(np.abs(me[i]-wA[i])))
        return diffs
    def info(self):        
        a=[self._1.W.array.shape, self._1.b.array.shape]
        b=[self._2.W.array.shape, self._2.b.array.shape]
        c=[self._3.W.array.shape, self._3.b.array.shape]
        d=[self._o.W.array.shape, self._o.b.array.shape]
        ash=(a[0][0]*a[0][1]) + (a[1][0])
        bsh=(b[0][0]*b[0][1]) + (b[1][0])
        csh=(c[0][0]*c[0][1]) + (c[1][0])
        dsh=(d[0][0]*d[0][1]) + (d[1][0])
        nos_params = ash+bsh+csh+dsh
        print('Layers-----------------')
        print(' Layer[1]\t',a,ash,'\n',
               'Layer[2]\t',b,bsh,'\n',
               'Layer[3]\t',c,csh,'\n',
               'Layer[o]\t',d,dsh,'\n',
              'Total Params:',nos_params)

"""
copy(mode: str = 'share') → chainer.link.Chain[source]
Copies the link hierarchy to new one.

The whole hierarchy rooted by this link is copied. There are three modes to perform copy. Please see the documentation for the argument mode below.

The name of the link is reset on the copy, since the copied instance does not belong to the original parent chain (even if exists).

Parameters
mode (str) – It should be either init, copy, or share. 

init means parameter variables under the returned link object is re-initialized by calling their initialize() method, so that all the parameters may have different initial values from the original link. 

copy means that the link object is deeply copied, so that its parameters are not re-initialized but are also deeply copied. Thus, all parameters have same initial values but can be changed independently. 

share means that the link is shallowly copied, so that its parameters’ arrays are shared with the original one. Thus, their values are changed synchronously. The default mode is share.

Returns
Copied link object.

Return type
Link

copyparams(link: chainer.link.Link, copy_persistent: bool = True) → None[source]
Copies all parameters from given link.

This method copies data arrays of all parameters in the hierarchy. The copy is even done across the host and devices. Note that this method does not copy the gradient arrays.

From v5.0.0: this method also copies the persistent values (e.g. the moving statistics of BatchNormalization). If the persistent value is an ndarray, the elements are copied. Otherwise, it is copied using copy.deepcopy(). The old behavior (not copying persistent values) can be reproduced with copy_persistent=False.

Parameters
link (Link) – Source link object.

copy_persistent (bool) – If True, persistent values are also copied. True by default.

count_params() → int[source]
Counts the total number of parameters.

This method counts the total number of scalar values included in all the Parameters held by this link and its descendants.

If the link containts uninitialized parameters, this method raises a warning.

Returns
The total size of parameters (int)

"""
