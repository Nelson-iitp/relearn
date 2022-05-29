

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  common.py :: shared/common procedures, environment validators, default keys and spaces
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
import gym.spaces
import os
import torch as tt
import torch.nn as nn
from io import BytesIO
import datetime
from math import floor
fake = lambda members: type('object', (object,), members)()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A] [space validation]
    > Only spaces accepted are Box and Discrete, both of these have (shape, dtype)
    > Discrete space always has shape = (), so ndim = 0, the dtype of discrete space is the default np.int size (int32 or int64)
    > Box space may have any shape including (), the dtype of box space can be set to any np.dtype
    
    > For observation_space, (Box spaces with ndim>0) are allowed
    > For action_space,      (Box spaces with ndim>0) or (Discrete space with ndim=0) are allowed
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def is_dis(space):
    return (type(space) is (gym.spaces.Discrete))

def is_box(space):
    return (type(space) is (gym.spaces.Box))

def is_valid_observation_space(space):
    return is_box(space) and (len(space.shape)>0)

def is_valid_action_space(space):
    is_box_space = is_box(space)
    is_dis_space = is_dis(space)
    is_dis_or_box = is_box_space or is_dis_space
    is_vector_if_box = ((len(space.shape)>0) if is_box_space else True)
    return is_dis_or_box and is_vector_if_box

def space_range(space):
    stype = int(is_dis(space)) - int(is_box(space))
    if stype == -1: # box
        return space.low, space.high
    elif stype == 1: # discrete
        return 0, space.n
    else:
        print(f'Cannot determine range of space [{space}] - invalid space-type.')
        return 0, 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B] [environment validation ]
    > env object should implement default gym.Env methods and variables
        > env.observation_space     (gym.space)
        > env.action_space          (gym.space)
        > env.reset()               :-> state
        > env.step(action)          :-> state, reward, done, info
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_env_attributes(env):
    required =  [ 'reset', 'step', 'observation_space', 'action_space' ]
    truth =     [hasattr(env, atr) for atr in required]
    return not (False in truth), {k:v for k,v in zip(required, truth)}

def check_env_spaces(env):
    required =   ['is_valid_observation_space',                         'is_valid_action_space'                  ]
    truth =      [ is_valid_observation_space(env.observation_space),    is_valid_action_space(env.action_space) ]
    return not (False in truth), {k:v for k,v in zip(required, truth)}

def check_env(env, verbose=True, caption="") -> None:
    attributes_truth, attributes_result =   check_env_attributes(env)
    spaces_truth, spaces_result =           check_env_spaces(env)
    if verbose:    
        results = {**attributes_result, **spaces_result}
        print(f'Env Check Results [{caption}]')
        for k,v in results.items():
            print(f'\t{k} :: [{v}]')
    return (attributes_truth and spaces_truth)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [C] [default keys and spaces]
    default keys in the spaces dictionary are (observation_key, action_key, reward_key, done_key )
    > The spaces dictionary is used by other components of the module
        > explorer objects use these keys to write to memory
        > memory objects implements key-based sampling used directly by algorithms
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

observation_key,    action_key,     reward_key,     done_key,    step_key    = \
'state',            'action',       'reward',       'done',       'step'

def default_spaces(observation_space, action_space):    
    # set default spaces - this is directly used by other components like explorer, memory and nets
    return {
        observation_key:    observation_space,
        action_key:         action_space,
        reward_key:         gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
        done_key:           gym.spaces.Box(low=0, high=1, shape=(), dtype=np.int8),
        step_key:           gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
    }



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [D] [torch.nn] 
    Some basic Neural Net models and helpers functions using torch.nn """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_models(dir_name, file_names, models):
    os.makedirs(dir_name, exist_ok=True)
    for θ, f in zip(models, file_names):
        tt.save(θ, os.path.join(dir_name, f))

def load_models(dir_name, file_names):
    return tuple( [tt.load(os.path.join(dir_name, f)) for f in file_names ])

def save_model(path, model):
    tt.save(model, path)

def load_model(path):
    return tt.load(path)

def clone_model(model, detach=False):
    """ use detach=True to sets the 'requires_grad' to 'False' on all of the parameters of the cloned model. """
    buffer = BytesIO()
    tt.save(model, buffer)
    buffer.seek(0)
    model_copy = tt.load(buffer)
    if detach:
        for p in model_copy.parameters():
            p.requires_grad=False
    model_copy.eval()
    del buffer
    return model_copy

def build_sequential(in_dim, layer_dims, out_dim, actF, actL ):
    layers = [nn.Linear(in_dim, layer_dims[0]), actF()]
    for i in range(len(layer_dims)-1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        layers.append(actF())
    layers.append(nn.Linear(layer_dims[-1], out_dim))
    _ = None if actL is None else layers.append(actL())
    return nn.Sequential( *layers )

class MLP(nn.Module):
    """ Multi layer Perceptron based parameterized networks for policy and value networks """
    def __init__(self, in_dim, layer_dims, out_dim, actF, actL):
        if len(layer_dims)<1:
            raise ValueError('need at least 1 layers')
        super(MLP, self).__init__()
        self.net = build_sequential(in_dim, layer_dims, out_dim, actF, actL )
    def forward(self, x):
        return self.net(x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [E] Policy Evaluation/Testing ~ does not use explorers """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@tt.no_grad()
def test_policy(env, pie, max_steps, verbose=0, render=0):
    """ test policy for one episode or end of mas_steps; returns (steps, return) """
    cs, act, reward, done, steps = env.reset(), 0, 0, False, 0
    total_reward = 0
    if verbose>0:
        print('[RESET]')
    if render==2:
        env.render()

    while not (done or steps>=max_steps):
        steps+=1
        act = pie.predict(cs)
        cs, reward, done, _ = env.step(act)
        total_reward+=reward

        if verbose>1:
            print('[STEP]:[{}], A:[{}], R:[{}], D:[{}], TR:[{}]'.format(steps, act, reward, done, total_reward))
        if render==3:
            env.render()

    if verbose>0:
        print('[TERM]: TS:[{}], TR:[{}]'.format(steps, total_reward))
    if render==1 or render==2:
        env.render()
    return steps, total_reward

def test_random(env, max_steps, verbose=0, render=0):
    """ test random policy (action_space.sample()) for one episode or end of mas_steps; returns (steps, return) """
    return test_policy(env = env, 
        pie = fake({'predict': lambda x, s:env.action_space.sample()}), 
        max_steps = max_steps, verbose=verbose, render=render)

def eval_policy(env, pie, max_steps, episodes, verbose=0, render=0, 
        verbose_result=True, render_result=True, 
        figsize=(16,8), caption="", return_fig=False):
    """ calls test_policy for multiple episodes; 
        returns results as pandas dataframe with cols: (#, steps, return) """
    from pandas import DataFrame
    test_hist = []
    for n in range(episodes):
        #print(f'\n-------------------------------------------------\n[Test]: {n}\n')
        result = test_policy(env, pie, max_steps, verbose=verbose, render=render)
        #print(f'steps:[{result[0]}], reward:[{result[1]}]')
        test_hist.append(result)
        
    test_hist=tt.as_tensor(test_hist)
    test_results = DataFrame(data = {
        '#' :       range(len(test_hist)),
        'steps'  :  test_hist[:, 0], 
        'return' :  test_hist[:, 1], 
        })
    if verbose_result:
        test_rewards = test_results['return']
        print(f'[Test Result]:\n\
        \tTotal-Episodes\t[{episodes}]\n\
        \tMean-Reward\t[{np.mean(test_rewards)}]\n\
        \tMedian-Reward\t[{np.median(test_rewards)}]\n\
        \tMax-Reward\t[{np.max(test_rewards)}]\n\
        \tMin-Reward\t[{np.min(test_rewards)}]\n\
        ')
        print(f'\n{test_results.describe()}\n')
    fig_return = plot_test_result(test_results, figsize=figsize, caption=caption, return_fig=return_fig) \
        if render_result else None
        
    return test_results, fig_return


def eval_random(env, max_steps, episodes, verbose=0, render=0, 
        verbose_result=True, render_result=True, 
        figsize=(16,8), caption="", return_fig=False):
    return eval_policy(
        env, 
        fake({'predict': lambda x, s:env.action_space.sample()}),
        max_steps, episodes, verbose=verbose, render=render, 
        verbose_result=verbose_result, render_result=render_result, 
        figsize=figsize, caption=caption, return_fig=return_fig
    )
        

def plot_test_result(val_res, figsize, caption, return_fig=False):
    xrange, val_hist_reward, val_hist_steps = val_res['#'], val_res['return'], val_res['steps']

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'[{caption}]')

    vrax = ax[0]
    vrax.plot(val_hist_reward, label='return', color='tab:green', linewidth=0.7)
    vrax.scatter(xrange, val_hist_reward, color='tab:green', marker='.')
    vrax.legend()

    vsax = ax[1]
    vsax.plot(val_hist_steps, label='steps', color='tab:purple', linewidth=0.7)
    vsax.scatter(xrange,  val_hist_steps, color='tab:purple', marker='.')
    vsax.legend()

    plt.show()

    return (fig if return_fig else None) #plot_validation_result(test_res, figsize, caption, return_fig=return_fig)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [Z] Misc """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class REMAP:
    def __init__(self,Input_Range, Mapped_Range) -> None:
        self.input_range(Input_Range)
        self.mapped_range(Mapped_Range)

    def input_range(self, Input_Range):
        self.Li, self.Hi = Input_Range
        self.Di = self.Hi - self.Li
    def mapped_range(self, Mapped_Range):
        self.Lm, self.Hm = Mapped_Range
        self.Dm = self.Hm - self.Lm
    def map2in(self, m):
        return ((m-self.Lm)*self.Di/self.Dm) + self.Li
    def in2map(self, i):
        return ((i-self.Li)*self.Dm/self.Di) + self.Lm

class REX(Exception):
    pass # basic exception for all classes

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Printing functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="", sep="|", end=""):
    """ returns a string representation of an array/list for printing """
    res=start
    for a in arr:
        res += (str(a) + sep)
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strD(arr, sep="\n", cep=":\n", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-={}DICT #[{}] : {}{}=-=-=-=-==-=-=-=-={}".format(sep, len(arr), caption, sep, sep)
    for k,v in arr.items():
        res+=str(k) + cep + str(v) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def show(x, cep='\t\t:', sw='__', ew='__', P = print):
    """ Note: 'sw' can accept tuples """
    for d in dir(x):
        if not (d.startswith(sw) or d.endswith(ew)):
            v = ""
            try:
                v = getattr(x, d)
            except:
                v='?'
            P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def showX(x, cep='\t\t:',P = print):
    """ same as showx but shows all members, skip startswith test """
    for d in dir(x):
        v = ""
        try:
            v = getattr(x, d)
        except:
            v='?'
        P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strU(form=["%Y","%m","%d","%H","%M","%S","%f"], start='', sep='', end=''):
    """ formated time stamp based UID ~ default form=["%Y","%m","%d","%H","%M","%S","%f"] """
    return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num:int, base:int, digs:int) -> list:
    """ convert base-10 integer (num) to base(base) array of fixed no. of digits (digs) """
    res = [ 0 for _ in range(digs) ]
    q = num
    for i in range(digs): # <-- do not use enumerate plz
        res[i]=q%base
        q = floor(q/base)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2int(num:list, base:int) -> int:
    """ convert array from given base to base-10  --> return integer """
    res = 0
    for i,n in enumerate(num):
        res+=(base**i)*n
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def reversed_cumulative_sum(arr): # ~ used to calculate 'rewards to go' 
    n, r = len(arr), np.zeros_like(arr)
    for i in reversed(range(n)):
        r[i] = arr[i] + (r[i+1] if i+1 < n else 0)
    return r
#-----------------------------------------------------------------------------------------------------



""" FOOT NOTE:

[ARCHIVE]


def plot_training_result(train_res, figsize, caption, return_fig=False):
    hist_loss, hist_return = train_res['loss'], train_res['return']

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'Training :[{caption}]')

    hlax = ax[0]
    hlax.plot(hist_loss, label='loss', color='tab:red', linewidth=0.7)
    hlax.legend()

    hrax = ax[1]
    hrax.plot(hist_return, label='return', color='tab:blue', linewidth=0.7)
    hrax.legend()
    plt.show()
    return (fig if return_fig else None)

"""
#-----------------------------------------------------------------------------------------------------
