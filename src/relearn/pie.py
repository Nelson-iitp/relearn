
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  pie.py :: Policy and Value representation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from .common import clone_model
import torch as tt
import torch.distributions as td

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A] Base Stohcastic Policy Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PIE: 
    """ Base class for Stohcastic Policy 
    
        NOTE: 'prediction_mode' arg is for the explorer to take action (calls with no_grad) 

        NOTE: base class should not be used directly, use inherited classes. 
                inherited class should implement:
                    ~ predict_deterministic         ~ for explorer
                    ~ predict_stohcastic            ~ for explorer
                    ~ __call__                      ~ in batch mode (called for calculating log-loss of distribution)
    
    
        NOTE: 
        member functions can be called in 
        
            ~ batch-mode - it means the input args is
                ~ a batch of states
                ~ tensors returned by a batch-sampling function
                ~ used with grad (for loss, learning)

            ~ explore-mode, it means input args is
                ~ a single state 
                ~ is numpy or int directly obtained from the environment's observation_space 
                ~ used with no_grad (for collecting experience)
    
    """

    def __init__(self, discrete_action, prediction_mode, has_target, dtype, device):
        # prediction_mode = True: deterministic, False:Distribution
       
        self.is_discrete = discrete_action # print('~ Use Categorical Policy')
        self.dtype, self.device = dtype, device
        self.has_target=has_target
        self.switch_prediction_mode(prediction_mode)

    def switch_prediction_mode(self, prediction_mode):
        # use action_mode = True for deterministic
        self.prediction_mode = prediction_mode
        self.predict = (self.predict_deterministic if prediction_mode else self.predict_stohcastic)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A.1]  Discrete Action Stohcastic Policy """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class dPIE(PIE):
    """ Discrete Action Stohcastic Policy : estimates categorical (softmax) distribution over output actions """

    def __init__(self, policy_theta, prediction_mode, has_target, dtype, device):
        super().__init__(True, prediction_mode, has_target,  dtype, device)

        # parameter setup
        self.theta = policy_theta.to(dtype=dtype, device=device) 
        self.parameters = self.theta.parameters
        # target parameter
        self.theta_ =( clone_model(self.theta, detach=True) if self.has_target else self.theta )
        # set to train=False
        self.theta.eval()
        self.theta_.eval()
        

    """ Policy output: distributional ~ called in batch mode"""
    def __call__(self, state): # returns categorical distribution over policy output 
        return td.Categorical( logits = self.theta(state) ) 
    
    def log_loss(self, state, action, weight):  # loss is -ve because need to perform gradient 'assent' on policy
        return -(  (self(state).log_prob(action) * weight).mean()  )


    """ Prediction: predict(state) used by explorer ~ called in explore mode"""
    def predict_stohcastic(self, state):
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self(state).sample().item()

    def predict_deterministic(self, state): 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self(state).probs.argmax().item() 

    def copy_target(self):
        if not self.has_target:
            return False
        self.theta_.load_state_dict(self.theta.state_dict())
        self.theta_.eval()
        return True

    def _save(self):
        #self.theta.is_discrete = self.is_discrete
        self.theta.has_target = self.has_target
        self.theta.dtype = self.dtype
        self.theta.device = self.device
        self.theta.prediction_mode = self.prediction_mode
        
    def save_(self):
        #self.theta.is_discrete = self.is_discrete
        del self.theta.has_target, self.theta.dtype, self.theta.device, self.theta.prediction_mode
        
    def save(pie, path):
        pie._save()
        tt.save(pie.theta, path)
        pie.save_()
        
    def load(path):
        theta = tt.load(path)
        pie = __class__(theta,  theta.prediction_mode, theta.has_target, theta.dtype, theta.device)
        pie.save_()
        return pie

    def train(self, mode):
        self.theta.train(mode)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A.2]  Continous Action Stohcastic Policy """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class cPIE(PIE):
    """ Continous Action Stohcastic Policy with one networks for Mean(loc = _mean) and 
            standalone Sdev(scale = _sdev) parameter (does not depend on state)
         : estimates Normal/Gausiian (diagonal) distribution over output actions """
    

    def get_policy_theta_scale(scale, action_shape):
        return tt.nn.Parameter(scale * tt.ones(action_shape)) #<-- NOTE: this is actually log(std_dev)
        
    def __init__(self, policy_theta_loc, policy_theta_scale, prediction_mode, has_target, dtype, device):
        """ here, policy_theta_scale is a number(float) - initial sigma =-0.5 """
        super().__init__(False, prediction_mode, has_target, dtype, device)
        # parameter setup
        self.theta_mean = policy_theta_loc.to(dtype=dtype, device=device)
        self.theta_sdev = policy_theta_scale.to(dtype=dtype, device=device)
        self.parameters_mean = self.theta_mean.parameters
        self.parameters_sdev = lambda: self.theta_sdev
        # target parameter
        self.theta_mean_ =( clone_model(self.theta_mean, detach=True) if self.has_target else self.theta_mean)
        self.theta_sdev_ =( ( self.theta_sdev.detach().clone() ) if self.has_target else self.theta_sdev)
        # set to train=False
        self.theta_mean.eval()
        self.theta_mean_.eval()


    """ Policy output: distributional ~ called in batch mode"""
    def __call__(self, state): # returns categorical distribution over policy output 
        return td.Normal( loc=self.theta_mean(state), scale=(self.theta_sdev.exp()) )
    
    def log_loss(self, state, action, weight):  # loss is -ve because need to perform gradient 'assent' on policy
        return -(  (self(state).log_prob(action).sum(axis=-1) * weight).mean()  )


    """ Prediction: predict(state) used by explorer ~ called in explore mode"""
    def predict_stohcastic(self, state):
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self(state).sample().cpu().numpy()

    def predict_deterministic(self, state): 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        # NOTE: should use - self(state).mean.cpu().numpy() 
        # #<--- but since we need mean only, no need to forward through sdev network
        return self.theta_mean(state).cpu().numpy()
        
    def copy_target(self):
        if not self.has_target:
            return False
        self.theta_mean_.load_state_dict(self.theta_mean.state_dict())
        self.theta_mean_.eval()
        self.theta_sdev_ = self.theta_sdev.detach().clone()
        return True

    def _save(self):
        #self.theta.is_discrete = self.is_discrete
        self.theta_mean.has_target = self.has_target
        self.theta_mean.dtype = self.dtype
        self.theta_mean.device = self.device
        self.theta_mean.prediction_mode = self.prediction_mode
        

    def save_(self):
        del self.theta_mean.has_target, self.theta_mean.dtype, \
            self.theta_mean.device, self.theta_mean.prediction_mode

    def save(pie, path_mean, path_sdev):
        pie._save()
        tt.save(pie.theta_mean, path_mean)
        tt.save(pie.theta_sdev, path_sdev)
        pie.save_()
        
    def load(path_mean, path_sdev):
        theta_mean = tt.load(path_mean)
        theta_sdev = tt.load(path_sdev)
        pie = __class__(theta_mean, theta_sdev, theta_mean.prediction_mode, 
                theta_mean.has_target, theta_mean.dtype, theta_mean.device)
        pie.save_()
        return pie
        
    def train(self, mode):
        self.theta_mean.train(mode)

        
class c2PIE(PIE):
    """ Continous Action Stohcastic Policy with seperate networks for Mean(loc = _mean) and Sdev(scale = _sdev) 
         : estimates Normal/Gausiian (diagonal) distribution over output actions """

    def __init__(self, policy_theta_loc, policy_theta_scale, prediction_mode, has_target, dtype, device):
        super().__init__(False, prediction_mode, has_target, dtype, device)

        # parameter setup
        self.theta_mean = policy_theta_loc.to(dtype=dtype, device=device)
        self.theta_sdev = policy_theta_scale.to(dtype=dtype, device=device) #<-- NOTE: this is actually log(std_dev)
        self.parameters_mean = self.theta_mean.parameters
        self.parameters_sdev = self.theta_sdev.parameters
        # target parameter
        self.theta_mean_ =( clone_model(self.theta_mean, detach=True) if self.has_target else self.theta_mean)
        self.theta_sdev_ =( clone_model(self.theta_sdev, detach=True) if self.has_target else self.theta_sdev)
        # set to train=False
        self.theta_mean.eval()
        self.theta_mean_.eval()
        self.theta_sdev.eval()
        self.theta_sdev_.eval()


    """ Policy output: distributional ~ called in batch mode"""
    def __call__(self, state): # returns categorical distribution over policy output 
        return td.Normal( loc=self.theta_mean(state), scale=(self.theta_sdev(state).exp()) )
    
    def log_loss(self, state, action, weight):  # loss is -ve because need to perform gradient 'assent' on policy
        return -(  (self(state).log_prob(action).sum(axis=-1) * weight).mean()  )


    """ Prediction: predict(state) used by explorer ~ called in explore mode"""
    def predict_stohcastic(self, state):
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self(state).sample().cpu().numpy()

    def predict_deterministic(self, state): 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        # NOTE: should use - self(state).mean.cpu().numpy() 
        # #<--- but since we need mean only, no need to forward through sdev network
        return self.theta_mean(state).cpu().numpy()
        
    def copy_target(self):
        if not self.has_target:
            return False
        self.theta_mean_.load_state_dict(self.theta_mean.state_dict())
        self.theta_sdev_.load_state_dict(self.theta_sdev.state_dict())
        self.theta_mean_.eval()
        self.theta_sdev_.eval()
        return True
        

    def _save(self):
        #self.theta.is_discrete = self.is_discrete
        self.theta_mean.has_target = self.has_target
        self.theta_mean.dtype = self.dtype
        self.theta_mean.device = self.device
        self.theta_mean.prediction_mode = self.prediction_mode
        
    def save_(self):
        del self.theta_mean.has_target, self.theta_mean.dtype, \
            self.theta_mean.device, self.theta_mean.prediction_mode

    def save(pie, path_mean, path_sdev):
        pie._save()
        tt.save(pie.theta_mean, path_mean)
        tt.save(pie.theta_sdev, path_sdev)
        pie.save_()
        
    def load(path_mean, path_sdev):
        theta_mean = tt.load(path_mean)
        theta_sdev = tt.load(path_sdev)
        pie = __class__(theta_mean, theta_sdev, theta_mean.prediction_mode, 
                theta_mean.has_target, theta_mean.dtype, theta_mean.device)
        pie.save_()
        return pie

    def train(self, mode):
        self.theta_mean.train(mode)
        self.theta_sdev.train(mode)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B] Base Value Netowrk Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class VAL: 
    """ base class for value estimators 
    
        NOTE: for Q-Values, underlying parameters value_theta should accept state-action pair as 2 sepreate inputs  
        NOTE: all Value functions (V or Q) are called in batch mode only """

    def __init__(self, value_theta,  discrete_action, has_target, dtype, device):
        self.dtype, self.device = dtype, device
        if discrete_action:
            self.is_discrete = True 
            self.call = self.call_discrete
            self.call_ = self.call_discrete_
        else:
            self.is_discrete = False

            self.call = self.call_continuous
            self.call_ = self.call_continuous_
        self.has_target = has_target
        self.theta = value_theta.to(dtype=dtype, device=device) 
        self.theta_ =( clone_model(self.theta, detach=True) if self.has_target else self.theta )
        self.parameters = self.theta.parameters
        # set to train=False
        self.theta.eval()
        self.theta_.eval()

    def __call__(self, state, target=False): #<-- called in batch mode
        return (self.call_(state) if target else self.call(state))

    def copy_target(self):
        if not self.has_target:
            return False
        self.theta_.load_state_dict(self.theta.state_dict())
        self.theta_.eval()
        return True

    def train(self, mode):
        self.theta.train(mode)
    
    def _save(self):
        self.theta.is_discrete = self.is_discrete
        self.theta.has_target = self.has_target
        self.theta.dtype = self.dtype
        self.theta.device = self.device

    def save_(self):
        del self.theta.is_discrete, self.theta.has_target, self.theta.dtype, self.theta.device

    def save(val, path):
        val._save()
        tt.save(val.theta, path)
        val.save_()
        

        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B.1]  State Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class sVAL(VAL): 
    # state-value function, same can be used for multi-Qvalue function based on output of value_theta
    def call_discrete(self, state): #<-- called in batch mode
        return tt.squeeze( self.theta ( state ), dim=-1 )

    def call_continuous(self, state): #<-- called in batch mode
        return tt.squeeze( self.theta ( state ), dim=-1 )

    def call_discrete_(self, state): #<-- called in batch mode
        return tt.squeeze( self.theta_ ( state ), dim=-1 )

    def call_continuous_(self, state): #<-- called in batch mode
        return tt.squeeze( self.theta_ ( state ), dim=-1 )

    def load(path):
        theta = tt.load(path)
        
        val = __class__(theta,  theta.is_discrete, theta.has_target, theta.dtype, theta.device)
        val.save_()
        return val
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B.2] State-Action Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class qVAL(VAL): 
    # state-action-value function 

    def call_continuous(self, state, action): #<-- called in batch mode
        return tt.squeeze( self.theta ( state, action  ), dim=-1 )
    
    def call_discrete(self, state, action):
        return tt.squeeze( self.theta ( state, action.unsqueeze(dim=-1) ), dim=-1 )

    def call_continuous_(self, state, action): #<-- called in batch mode
        return tt.squeeze( self.theta_ ( state, action  ), dim=-1 )
    
    def call_discrete_(self, state, action):
        return tt.squeeze( self.theta_ ( state, action.unsqueeze(dim=-1) ), dim=-1 )

    def __call__(self, state, action, target=False): #<-- called in batch mode
        return (self.call_(state, action) if target else self.call(state, action))

    def load(path):
        theta = tt.load(path)
        
        val = __class__(theta,  theta.is_discrete, theta.has_target, theta.dtype, theta.device)
        val.save_()
        return val
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B.3]  Multi-Q Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class mVAL(VAL): 
    # state-value function, same can be used for multi-Qvalue function based on output of value_theta
    def call_discrete(self, state): #<-- called in batch mode
        return self.theta ( state )

    def call_continuous(self, state): #<-- called in batch mode
        return  self.theta ( state )

    def call_discrete_(self, state): #<-- called in batch mode
        return self.theta_ ( state )

    def call_continuous_(self, state): #<-- called in batch mode
        return self.theta_ ( state )

    def predict(self, state): # <---- called in explore mode
        # works for discrete action and multi-Qvalue only 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self.theta ( state ).argmax().item()
        
    def load(path):
        theta = tt.load(path)
        
        val = __class__(theta,  theta.is_discrete, theta.has_target, theta.dtype, theta.device)
        val.save_()
        return val

#-----------------------------------------------------------------------------------------------------
""" FOOT NOTE:

[Policy Representation]

Policy representation comes into question only when we talk about policy-learning methods,
we do not need explicit policy when dealing with value learning, where policy is derived based on value

Types of policy : 


    [1] Based on timestep

    > Non-Stationary: 
        ~ depends on timestep (TODO: make a col in memory for timestep)
        ~ useful in finite horizon context
        ~ here the cumulative reward is limited by finite number of future timesteps

    > Stationary:
        ~ does not depend on timestep
        ~ used in infinite horizon context 
        ~ cumulative reward is limited by a choice of 'discount factor'

    [2] Based on stohcasticity

    > Deterministic
        ~ outputs the exact action
    
    > Stohcastic
        ~ outputs a distribution over actions

    
    NOTE: based on action space type, we can have different types of parameterized policies:
        > Deterministic policy for Discrete Action spaces   ~   Not possible
        > Deterministic policy for Box Action spaces        ~   out puts the action vetor
        > Stohcastic policy for Discrete Action spaces      ~   outputs Categorical Dist
        > Stohcastic policy for Box Action spaces           ~ outputs Normal Dist


[Distributional DQN]
    Estimates a 'Value Distribution' 
        ~ the distribution is limited within a range (Vmax-Vmin)
        ~ requires knowledge of Vmax and Vmin apriori
        ~ also required the number of bins, as the value for each state (or state-action if using Q-function)
            is divided over a fixed range - into fixed number of bins
        ~ The value distribution Zπ is a mapping from state-action pairs to distributions of returns when following policy π
        ~ the 'C51' algorithm uses 51 bins for the purpose, which is found emperically

[ARCHIVE]
def optim(self, optim_name, optim_args, lrs_name, lrs_args):
    self.opt = optim_name( self.theta.parameters(), **optim_args)
    self.lrs = (lrs_name(self.opt, **lrs_args) if lrs_name else None)

def zero_grad(self):
    self.opt.zero_grad()

def step(self):
    self.step_optim()
    self.step_lrs()

def step_optim(self):
    return self.opt.step()
    
def step_lrs(self):
    return (self.lrs.step() if self.lrs else None)
"""
#-----------------------------------------------------------------------------------------------------