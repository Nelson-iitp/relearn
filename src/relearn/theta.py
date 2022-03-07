#-----------------------------------------------------------------------------------------------------
# relearn/theta.py
#-----------------------------------------------------------------------------------------------------
import torch, os
import torch.nn as nn
import numpy as np
#from .core import SPACE

#-----------------------------------------------------------------------------------------------------
def strScalar(S):
    return str(S.item())
def strVector(S):
    res=""
    for ti in S.flatten():
        res+=str(ti.item())+","
    return res
def compare_weights(w1, w2):
    for i,(k1,k2) in enumerate(zip(w1,w2)):
        #print(i,k1,k2, w1[k1]-w2[k2])
        if not torch.equal(w1[k1],w2[k2]):
            return False
    return True
#-----------------------------------------------------------------------------------------------------
FROOT = '_'

class SSFUNC:
    """ space -> space  single-domain + single-range function in a table """

    def __init__(self, device, in_space, out_space):
        
        self.in_space, self.out_space, self.device  = in_space, out_space, device         
        self.in_map =          strScalar if in_space.scalar else strVector
        self.out_default =     self.out_space.zeros(self.device)
        self.F = {}
        self.in_count = 1
        self.out_count = 1

    def acall(self, X):
        sx = self.in_map(X)
        if not sx in self.F:
            self.F[sx] = self.out_space.zeros(self.device)
        return self.F[sx]

    def call(self, X):
        sx = self.in_map(X)
        return self.F[sx] if sx in self.F else self.out_default

class MSFUNC:
    """ (space,...) -> space  multi-domain + single-range function in a table """

    def __init__(self, device, in_spaces, out_space):
        
        self.in_spaces, self.out_space, self.device  = in_spaces, out_space, device         
        self.in_maps =         [ (strScalar if in_space.scalar else strVector) for in_space in self.in_spaces]
        self.out_default =     self.out_space.zeros(self.device)
        self.F = {}
        self.in_count = len(self.in_spaces)
        self.out_count = 1

    def acall(self, *X):
        assert (len(X) == self.in_count)
        F = self.F
        for i,x in enumerate(X):
            sx = self.in_maps[i](x)
            if not sx in F:
                F[sx] = {}
            F = F[sx]

        if not FROOT in F:
            F[FROOT] = self.out_space.zeros(self.device)
        return F[FROOT]

    def dcall(self, *X):
        assert (len(X) <= self.in_count)
        F = self.F
        found = True
        for i,x in enumerate(X):
            sx = self.in_maps[i](x)
            if not sx in F:
                found=False
                break
            else:
                F = F[sx]
        return F if found else {}

    def call(self, *X):
        assert (len(X) == self.in_count)
        F = self.F
        found = True
        for i,x in enumerate(X):
            sx = self.in_maps[i](x)
            if not sx in F:
                found=False
                break
            else:
                F = F[sx]
        return F[FROOT] if found else self.out_default

class SMFUNC:
    """ space -> (space,...)  single-domain + multi-range function in a table """

    def __init__(self, device, in_space, out_spaces):
        
        self.in_space, self.out_spaces, self.device  = in_space, out_spaces, device         
        self.in_map =          strScalar if self.in_space.scalar else strVector
        self.out_defaults =     [  out_space.zeros(self.device) for out_space in self.out_spaces ]
        self.F = {}
        self.in_count = 1
        self.out_count = len(self.out_spaces)

    def acall(self, X):
        sx = self.in_map(X)
        if not sx in self.F:
            self.F[sx] = [  out_space.zeros(self.device) for out_space in self.out_spaces ]
        return self.F[sx]

    def call(self, X):
        sx = self.in_map(X)
        return self.F[sx] if sx in self.F else self.out_defaults

class MMFUNC:
    """ (space,...) -> (space,...)  multi-domain + multi-range function in a table """

    def __init__(self, device, in_spaces, out_spaces):
        
        self.in_spaces, self.out_spaces, self.device  = in_spaces, out_spaces, device         
        self.in_maps =          [ (strScalar if in_space.scalar else strVector) for in_space in self.in_spaces]
        self.out_defaults =     [  out_space.zeros(self.device) for out_space in self.out_spaces ]
        self.F = {}
        self.in_count = len(self.in_spaces)
        self.out_count = len(self.out_spaces)

    def acall(self, *X):
        assert (len(X) == self.in_count)
        F = self.F
        for i,x in enumerate(X):
            sx = self.in_maps[i](x)
            if not sx in F:
                F[sx] = {}
            F = F[sx]

        if not FROOT in F:
            F[FROOT] = [  out_space.zeros(self.device) for out_space in self.out_spaces  ]
        return F[FROOT]

    def dcall(self, *X):
        assert (len(X) <= self.in_count)
        F = self.F
        found = True
        for i,x in enumerate(X):
            sx = self.in_maps[i](x)
            if not sx in F:
                found=False
                break
            else:
                F = F[sx]
        return F if found else {}

    def call(self, *X):
        assert (len(X) == self.in_count)
        F = self.F
        found = True
        for i,x in enumerate(X):
            sx = self.in_maps[i](x)
            if not sx in F:
                found=False
                break
            else:
                F = F[sx]
        return F[FROOT] if found else self.out_defaults


class NETFUNC(nn.Module):
    """ space -> space function as a multi layer preceptron """

    def __init__(self, device, in_space, out_space, layer_dims, actF, dtype=torch.float32):
        super(NETFUNC, self).__init__()

        self.in_space, self.out_space, self.device = in_space, out_space, device
        self.net_dtype = dtype
        self.n_layers = len(layer_dims)
        if self.n_layers<1:
            raise ValueError('need at least 1 layers')
        layers = [ nn.Linear(self.in_space.nflat, layer_dims[0], dtype=self.net_dtype, device=self.device), actF() ]
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], dtype=self.net_dtype, device=self.device))
            layers.append(actF())
        layers.append(nn.Linear(layer_dims[-1], self.out_space.nflat, dtype=self.net_dtype, device=self.device))
        # do not add activation on last layer
        self.SEQL = nn.Sequential( *layers )
        self.n_layers+=1

    # call<i0>
    def _call00(self, x):
        # only reshapes input and output
        return self.SEQL(x.reshape(1, self.in_space.nflat))[0].reshape(self.out_space.shape)
    def _call01(self, x):
        # reshapes input and output, convert output dtype
        return self.SEQL(x.reshape(1, self.in_space.nflat))[0].reshape(self.out_space.shape).to(self.out_space.dtype)
    def _call10(self, x):
        # reshapes input and output, changes input dtype and device
        return self.SEQL(x.reshape(1, self.in_space.nflat).to(dtype=self.net_dtype, device=self.device))[0].reshape(self.out_space.shape)
    def _call11(self, x):
        # reshapes input and output, changes input dtype and device, changes output dtype
        return self.SEQL(x.reshape(1, self.in_space.nflat).to(dtype=self.net_dtype, device=self.device ))[0].reshape(self.out_space.shape).to(self.out_space.dtype)
    
    def callx(self, x, i=True, o=False): #<-- default call10()
        X = x.reshape(1, self.in_space.nflat).to(dtype=self.net_dtype, device=self.device) if i else x.reshape(1, self.in_space.nflat)
        return self.SEQL(X)[0].reshape(self.out_space.shape).to(self.out_space.dtype) if o else self.SEQL(X)[0].reshape(self.out_space.shape)
    def callX(self, X, i=True, o=False): #<-- default call10()
        XX = X.reshape(-1, self.in_space.nflat).to(dtype=self.net_dtype, device=self.device) if i else X.reshape(-1, self.in_space.nflat)
        return self.SEQL(XX).reshape((-1,)+self.out_space.shape).to(self.out_space.dtype) if o else self.SEQL(XX).reshape((-1,)+self.out_space.shape)

    def forward(self, X):
        return self.SEQL(X)

    def info(self, cap="", P=print):
        P('--------------------------')
        P(cap, 'No. Layers:\t', self.n_layers)
        P('--------------------------')
        # print(2*(pie.Q.n_layers+1)) <-- this many params
        std = self.state_dict()
        total_params = 0
        for param in std:
            nos_params = torch.prod(torch.tensor(std[param].shape),0).item() #shape2size(std[param].shape)
            P(param, '\t', nos_params, '\t', std[param].shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params
    

    def save_external(self, path):
        torch.save(self.state_dict(), path)
        
    def load_external(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

#-----------------------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Multi-Layer Perceptron
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class MLP:
    """ Implements custom multi layer perceptron """

    def tensor(self, data, rgrad=True):
        return torch.tensor(data, device=self.device, dtype=self.dtype, requires_grad=rgrad)

    def __init__(self, layer_dims, device, dtype, actF, init_range=(-0.001, 0.001), seed=None, from_param=False):
        self.device, self.dtype = device, dtype
        self.rng = np.random.default_rng(seed)
        self.init_low, self.init_high = init_range
        
        if from_param:
            self.parameters = layer_dims
            self.acts = []
            self.n_layers = 0

            for i in range(1, int(len(self.parameters)/2)):
                self.acts.append(actF())
                self.acts.append(None)
                self.n_layers +=1
        else:
            self.parameters = [] 
            self.acts = []
            self.n_layers = 0

            for i in range(1, len(layer_dims)):
                self.parameters.append( self.tensor(self.rng.uniform(self.init_low, self.init_high, size=(layer_dims[i], layer_dims[i-1]))) )
                self.parameters.append( self.tensor(self.rng.uniform(self.init_low, self.init_high, size=(layer_dims[i]))) )
                self.acts.append(actF())
                self.acts.append(None)
                self.n_layers +=1
        
    def callx(self, x):
        z = x.flatten()
        for l in range(0, len(self.parameters)-2, 2):
            z = self.acts[l]( self.parameters[l] @ z + self.parameters[l+1] ) 
        logits = self.parameters[-2] @ z + self.parameters[-1] 
        return logits

    def callX(self, X):
        Y= []
        for x in X:
            Y.append(self.callx(x))
        return torch.stack(Y)


    def step_grad(self, lr):
        with torch.no_grad():
            for p in self.parameters:
                p -= lr*p.grad

    def zero_grad(self):
        with torch.no_grad():
            for p in self.parameters:
                p.grad = None
            
    def info(self, show_vals=False, P=print):
        P('--------------------------')
        P('~ N_LAYERS:[{}]\n~ D_TYPE:[{}]\n~ DEV:[{}]'.format(self.n_layers, self.dtype, self.device))
        P('--------------------------')
        total_params = 0
        for px,param in enumerate(self.parameters):
            nos_params = torch.prod(torch.tensor(param.shape),0).item()
            if px%2 == 0:
                P('--> Weights[{}]:: Params[{}] of Shape[{}]'.format(int(px/2), nos_params,  param.shape))
            else:
                P('--> Bias[{}]:: Params[{}] of Shape[{}]'.format(int(px/2), nos_params,  param.shape))
            if show_vals:
                P(' ~--> [PARAMETER TENSOR]:', param)
            #P(px, '\t', nos_params, '\t', param.shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params

    def copy_weights(self, M):
        with torch.no_grad():
            for to_layer, from_layer in zip(self.parameters, M.parameters):
                to_layer.data.copy_(from_layer.data)
        return
    def diff_weights(self, M):
        diff = []
        with torch.no_grad():
            for to_layer, from_layer in zip(self.parameters, M.parameters):
                diff.append( to_layer - from_layer )
        return diff

    def save_external(self, path):
        os.makedirs(path, exist_ok=True)
        with torch.no_grad():
            for l,layer in enumerate(self.parameters):
                np.save(os.path.join(path, str(l)+'.npy'), layer.data.numpy() )
        return
        
    def load_external(self, path):
        with torch.no_grad():
            for l,layer in enumerate(self.parameters):
                layer.data.copy_(self.tensor( np.load(os.path.join(path, str(l)+'.npy')),rgrad=False))
                layer.grad = None  
        #self.zero_grad()
        return


    def load_net(self, netf):
        with torch.no_grad():
            paramf = netf.parameters()
            for ps, pf in zip(self.parameters, paramf):
                ps.copy_(pf)

#-----------------------------------------------------------------------------------------------------
# Foot-Note:
"""
NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------