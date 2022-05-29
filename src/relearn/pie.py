
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  pie.py :: Policy and Value representation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from .common import is_dis, is_box, REX, clone_model
import torch as tt
import torch.distributions as td



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A] Base Stohcastic Policy Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class π: 
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

    def __init__(self, action_space, prediction_mode, has_target, dtype, device):
        # prediction_mode = True: deterministic, False:Distribution
        if (is_dis(action_space)):
            self.is_discrete = True # print('~ Use Categorical Policy')
        elif (is_box(action_space)):
            self.is_discrete = False # print('~ Use Diagonal Gaussian Policy')
        else:
            raise REX(f'Invalid action space for policy :[{action_space}]')
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
class dPIE(π):
    """ Discrete Action Stohcastic Policy : estimates categorical (softmax) distribution over output actions """

    def __init__(self, policy_theta, action_space, prediction_mode, has_target, dtype, device):
        super().__init__(action_space, prediction_mode, has_target,  dtype, device)
        if not self.is_discrete:
            raise REX(f'Invalid action space for discrete policy :[{action_space}]')
        
        # parameter setup
        self.θ = policy_theta.to(dtype=dtype, device=device) 
        self.parameters = self.θ.parameters
        # target parameter
        self.θ_ =( clone_model(self.θ, detach=True) if self.has_target else self.θ )
        # set to train=False
        self.θ.eval()
        self.θ_.eval()
        

    """ Policy output: distributional ~ called in batch mode"""
    def __call__(self, state): # returns categorical distribution over policy output 
        return td.Categorical( logits = self.θ(state) ) 
    
    def log_loss(self, state, action, weight):  # loss is -ve because need to perform gradient 'assent' on policy
        return -(  (self(state).log_prob(action) * weight).mean()  )


    """ Prediction: predict(state) used by explorer ~ called in explore mode"""
    def predict_stohcastic(self, state):
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self(state).sample().item()

    def predict_deterministic(self, state): 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self(state).argmax().item() 

    def copy_target(self):
        if not self.has_target:
            return False
        self.θ_.load_state_dict(self.θ.state_dict())
        self.θ_.eval()
        return True



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A.2]  Continous Action Stohcastic Policy """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class cPIE(π):
    """ Continous Action Stohcastic Policy with one networks for Mean(loc = μ) and 
            standalone Sdev(scale = σ) parameter (does not depend on state)
         : estimates Normal/Gausiian (diagonal) distribution over output actions """

    def __init__(self, policy_theta_loc, policy_theta_scale, action_space, prediction_mode, has_target, dtype, device):
        """ here, policy_theta_scale is a number(float) - initial sigma =-0.5 """
        super().__init__(action_space, prediction_mode, has_target, dtype, device)
        if self.is_discrete:
            raise REX(f'Invalid action space for continuous policy :[{action_space}]')

        # parameter setup
        self.θμ = policy_theta_loc.to(dtype=dtype, device=device)
        self.θσ = tt.nn.Parameter(policy_theta_scale * tt.ones(action_space.shape, dtype=self.float32, device=self.device)) #<-- NOTE: this is actually log(std_dev)
        self.parameters_mean = self.θμ.parameters
        self.parameters_sdev = lambda: self.θσ
        # target parameter
        self.θμ_ =( clone_model(self.θμ, detach=True) if self.has_target else self.θμ)
        self.θσ_ =( ( self.θσ.detach().clone() ) if self.has_target else self.θσ)
        # set to train=False
        self.θμ.eval()
        self.θμ_.eval()


    """ Policy output: distributional ~ called in batch mode"""
    def __call__(self, state): # returns categorical distribution over policy output 
        return td.Normal( loc=self.θμ(state), scale=(self.θσ.exp()) )
    
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
        return self.θμ(state).cpu().numpy()
        
    def copy_target(self):
        if not self.has_target:
            return False
        self.θμ_.load_state_dict(self.θμ.state_dict())
        self.θμ_.eval()
        self.θσ_ = self.θσ.detach().clone()
        return True

class c2PIE(π):
    """ Continous Action Stohcastic Policy with seperate networks for Mean(loc = μ) and Sdev(scale = σ) 
         : estimates Normal/Gausiian (diagonal) distribution over output actions """

    def __init__(self, policy_theta_loc, policy_theta_scale, action_space, prediction_mode, has_target, dtype, device):
        super().__init__(action_space, prediction_mode, has_target, dtype, device)
        if self.is_discrete:
            raise REX(f'Invalid action space for continuous policy :[{action_space}]')

        # parameter setup
        self.θμ = policy_theta_loc.to(dtype=dtype, device=device)
        self.θσ = policy_theta_scale.to(dtype=dtype, device=device) #<-- NOTE: this is actually log(std_dev)
        self.parameters_mean = self.θμ.parameters
        self.parameters_sdev = self.θσ.parameters
        # target parameter
        self.θμ_ =( clone_model(self.θμ, detach=True) if self.has_target else self.θμ)
        self.θσ_ =( clone_model(self.θσ, detach=True) if self.has_target else self.θσ)
        # set to train=False
        self.θμ.eval()
        self.θμ_.eval()
        self.θσ.eval()
        self.θσ_.eval()


    """ Policy output: distributional ~ called in batch mode"""
    def __call__(self, state): # returns categorical distribution over policy output 
        return td.Normal( loc=self.θμ(state), scale=(self.θσ(state).exp()) )
    
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
        return self.θμ(state).cpu().numpy()
        
    def copy_target(self):
        if not self.has_target:
            return False
        self.θμ_.load_state_dict(self.θμ.state_dict())
        self.θσ_.load_state_dict(self.θσ.state_dict())
        self.θμ_.eval()
        self.θσ_.eval()
        return True
        



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B] Base Value Netowrk Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class φ: 
    """ base class for value estimators 
    
        NOTE: for Q-Values, underlying parameters value_theta should accept state-action pair as 2 sepreate inputs  
        NOTE: all Value functions (V or Q) are called in batch mode only """

    def __init__(self, value_theta,  action_space, has_target, dtype, device):
        self.dtype, self.device = dtype, device
        if (is_dis(action_space)):
            self.is_discrete = True 
            self.discrete_action_range = tt.arange(0, action_space.n, 1).to(dtype=dtype, device=device) 
            self.call = self.call_continuous
            self.call_ = self.call_continuous_
        elif (is_box(action_space)):
            self.is_discrete = False
            self.call = self.call_discrete
            self.call_ = self.call_discrete_
        else:
            raise REX(f'Invalid action space for policy :[{action_space}]')
        self.has_target = has_target
        self.θ = value_theta.to(dtype=dtype, device=device) 
        self.θ_ =( clone_model(self.θ, detach=True) if self.has_target else self.θ )
        self.parameters = self.θ.parameters
        # set to train=False
        self.θ.eval()
        self.θ_.eval()

    def __call__(self, state, target): #<-- called in batch mode
        return self.call_(state) if target else self.call(state)

    def copy_target(self):
        if not self.has_target:
            return False
        self.θ_.load_state_dict(self.θ.state_dict())
        self.θ_.eval()
        return True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B.1]  State Value, Multi-Q Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class sVAL(φ): 
    # state-value function, same can be used for multi-Qvalue function based on output of value_theta
    def call_discrete(self, state): #<-- called in batch mode
        return tt.squeeze( self.θ ( state ), dim=-1 )

    def call_continuous(self, state): #<-- called in batch mode
        return tt.squeeze( self.θ ( state ), dim=-1 )

    def call_discrete_(self, state): #<-- called in batch mode
        return tt.squeeze( self.θ_ ( state ), dim=-1 )

    def call_continuous_(self, state): #<-- called in batch mode
        return tt.squeeze( self.θ_ ( state ), dim=-1 )

    def predict(self, state): # <---- called in explore mode
        # works for discrete action and multi-Qvalue only 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self.θ ( state ).argmax().item()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B.2] State-Action Value Network """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class qVAL(φ): 
    # state-action-value function 

    def call_continuous(self, state, action): #<-- called in batch mode
        return tt.squeeze( self.θ ( state, action  ), dim=-1 )
    
    def call_discrete(self, state, action):
        return tt.squeeze( self.θ ( state, action.unsqueeze_(dim=-1) ), dim=-1 )

    def call_continuous_(self, state, action): #<-- called in batch mode
        return tt.squeeze( self.θ_ ( state, action  ), dim=-1 )
    
    def call_discrete_(self, state, action):
        return tt.squeeze( self.θ_ ( state, action.unsqueeze_(dim=-1) ), dim=-1 )

    def predict(self, state): # <---- called in explore mode
        # works for discrete action 
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self.θ ( state, self.discrete_action_range ).argmax().item()
       


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
    self.opt = optim_name( self.θ.parameters(), **optim_args)
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