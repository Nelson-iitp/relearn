# [module] ~ R E L E A R N ~
print(
"""
                       ~ R E L E A R N ~
""")


#-----------------------------------------------------------------------------------------------------
# [relearn.common]
import relearn.common as common
#-----------------------------------------------------------------------------------------------------

# [A] [space validation], [B] [environment validation ]
from relearn.common import is_dis, is_box, space_range, check_env

# [C] [default keys and spaces]
from relearn.common import observation_key, action_key, reward_key, done_key, step_key, default_spaces

# [D] [torch.nn]
from relearn.common import save_model, save_models, load_model, load_models, clone_model
from relearn.common import MLP, DLP, MLP2

# [E]  Policy Evaluation/Testing
from relearn.common import test_policy, test_random, eval_policy, eval_random
from relearn.common import validate_episode, validate_episodes


# [z] Misc
from relearn.common import REX, REMAP
from relearn.common import strA, strD, strU, show, showX



#-----------------------------------------------------------------------------------------------------
# [relearn.pie]
import relearn.pie as pie
#-----------------------------------------------------------------------------------------------------
from relearn.pie import dPIE, cPIE, c2PIE
from relearn.pie import sVAL, qVAL, mVAL
#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
# [relearn.exp]
import relearn.exp as exp
#-----------------------------------------------------------------------------------------------------
from relearn.exp import MEM, EXP, make_exp, make_mem, ExploreMode
#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
# [relearn.mdp]
import relearn.mdp as mdp
#-----------------------------------------------------------------------------------------------------
from relearn.mdp import treeMDP, randMDP
#-----------------------------------------------------------------------------------------------------


import relearn.core as core
import relearn.core.dqn as dqn
import relearn.core.spg as spg
import relearn.core.vpg as vpg


""" FOOT NOTE:

[ Symbols ]

 α β γ δ ε ζ η θ κ λ μ ξ π ρ ς σ φ ψ ω τ ϐ ϑ ϕ Ω ℓ Λ Γ Θ ϴ Φ Ψ Ξ Δ 



"""
#-----------------------------------------------------------------------------------------------------