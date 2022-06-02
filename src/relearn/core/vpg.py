""" VPG - Vanilla Policy Gradients

    ~ implements policy gradients with 2 types of advantage estimate ('use_q_values' argument)
        ~ Q(s,a)
        ~ Rewards-2-Go - V(s) <-- here b(s)=V(s)

    ~ NOTE: can use State-Value and State-Action-Value function for advantage 

"""

from math import inf, nan
import torch as tt
import numpy as np
#import torch.optim as oo
import torch.nn as nn
import matplotlib.pyplot as plt
from ..pie import dPIE, cPIE, sVAL, qVAL
from ..exp import make_exp, ExploreMode
from ..common import validate_episodes, validate_episode, is_dis, reversed_cumulative_sum, REX
from ..common import observation_key, action_key, reward_key
# first need a PIE wrapping a theta, this can either be -
#    discrete-action Categorical policy
#    continuous-action diagonal Gaussian policy
def get_dis_pie(policy_theta, has_target, dtype, device):
    return dPIE(policy_theta, prediction_mode=False, has_target=has_target, dtype=dtype, device=device)
def get_box_pie(policy_theta, action_shape, has_target, dtype, device):
    return cPIE(policy_theta_loc=policy_theta,
                policy_theta_scale=cPIE.get_policy_theta_scale(-0.5, action_shape),
                prediction_mode=False,
                has_target=has_target,
                dtype=dtype, device=device)

def get_val(use_q_values, value_theta, discrete_action, has_target, dtype, device):
    if use_q_values:
        return qVAL(value_theta, discrete_action, has_target, dtype, device)
    else:
        return sVAL(value_theta, discrete_action, has_target, dtype, device)

def get_exp(env, memory_capacity, memory_seed, extra_spaces): 
    # on policy explorer
    return make_exp(
        env=env,
        memory_capacity=memory_capacity,
        memory_seed=memory_seed,
        **extra_spaces) #<---- note this returns exp, mem


def train(
        policy_theta, 
        value_theta,
        dtype, 
        device, 
        env,
        gamma,
        memory_capacity, 
        memory_seed, 
        extra_spaces,
        optFun, 
        optArg, 
        lrsFun, 
        lrsArgs,
        voptFun, 
        voptArg, 
        vlrsFun, 
        vlrsArgs,
        viters,
        batch_size,
        epochs,
        verbf,
        plot_results,
        validations_envs, 
        validation_freq, 
        validation_episodes,
        validation_verbose, 
        validation_render, 
        validation_deteministic,
        save_as,
        use_q_values

        ):
    
    # NOTE: Do we want to use gamma? - makes the process very slow
    use_gamma = (gamma>0.0)

    # setup policy
    has_target = False # does not use any target
    discrete_action = is_dis(env.action_space)

    pie = (get_dis_pie(policy_theta, has_target, dtype, device)) \
            if discrete_action else \
          (get_box_pie(policy_theta, env.action_space.shape, has_target, dtype, device))

    val = get_val(use_q_values, value_theta, discrete_action, has_target, dtype, device)

    # setup explorer and memory
    exp, mem = get_exp(env, memory_capacity, memory_seed, extra_spaces)
    exp.reset(clear_memory=False, episodic=True) #<-- monete-carlo means episodic
    exp.mode(ExploreMode.policy, pie, args=None)
    # setup optimizer
    opt = (optFun(pie.parameters(), **optArg)) \
             if discrete_action else \
          (optFun((*pie.parameters_mean(), pie.parameters_sdev()), **optArg))
    lrs = lrsFun(opt, **lrsArgs)

    vopt = voptFun(val.parameters(), **voptArg)
    vlrs = vlrsFun(vopt, **vlrsArgs)
    voss = nn.MSELoss()


    do_validate = ((len(validations_envs)>0) and validation_freq and validation_episodes)
    mean_validation_return, mean_validation_steps = nan, nan
    # ===== One Training Epoch ====== #
    train_hist, validation_hist = [], []
    for epoch in range(epochs):
        epoch_ratio = epoch/epochs
        if (epoch%verbf==0):
            print(f'[{(100*epoch_ratio):.2f} %]')
        # ------------------- ------------------- 
        # prepare a bacth of trajectories (D)
        # ------------------- ------------------- 
        D = []
        for _ in range(batch_size):
            exp.clear_snap()          # clear previous episode (one episode stores for each in range(batch_size))
            ne = exp.explore(1)        # explore single episode / trajectory
            
            if (exp.memory.at_max):
                # to debug this use ---> exp.memory.render_all()
                raise REX(f'[!] MEMORY-ERROR: [ptr:{exp.memory.ptr}] [max:{exp.memory.at_max}] [count:{exp.memory.count()}]')

            # read trajectory indices in memory (read all indices since there is only one episode)
            n, samples = exp.memory.sample_all_() #<-- dont use [tt.arange(0, memory.ptr, 1)] as we do not require the terimal state 
            if (n!=ne):
                raise REX(f'[!] EXPLORER-ERROR: [timesteps:{n}] do not match [explored:{ne}]')
            # read into a dict
            trajectory = exp.memory.readkeis(
                    (samples-1,                samples,                 samples               ), 
                    (observation_key,          action_key,              reward_key            ),
                    ('cS',                     'A',                     'R'                   ))
            
            # compute additional stuff required for policy update
            #assert(ne == len(trajectory['R']))
            #assert(gamma_array.shape == trajectory['R'].shape)
            trajectory.update({'N': ne})        # no of steps (episode-len)
            trajectory.update({'RET': np.sum(trajectory['R']) })  # return or total reward of the episode 
            if use_gamma: # discounted rewards
                trajectory.update({'dR':           
                    trajectory['R'] * np.logspace(start=1, stop=ne, num=ne, endpoint=True, base=gamma)  }) 
                trajectory.update({'dRET': np.sum(trajectory['dR']) })#<-- discounted sum needed
            else:
                trajectory.update({'dR': trajectory['R']})
                trajectory.update({'dRET': trajectory['RET']})
                 
            trajectory.update({'R2G': reversed_cumulative_sum(trajectory['dR']) }) # Value of each state

            D.append(trajectory) # add to batch
        
        # stack up into a single batch to pass to the loss function - also convert to tensors
        batch_R2G = tt.tensor(np.hstack( tuple( [D[i]['R2G']    for i in range(batch_size)] ) ), dtype=dtype, device=device)
        batch_S = tt.tensor(np.vstack( tuple( [D[i]['cS']   for i in range(batch_size)] ) ), dtype=dtype, device=device)
        batch_A = \
            tt.tensor(np.hstack( tuple( [D[i]['A']    for i in range(batch_size)] ) ), dtype=dtype, device=device) \
                if (discrete_action) else \
            tt.tensor(np.vstack( tuple( [D[i]['A']    for i in range(batch_size)] ) ), dtype=dtype, device=device) 
        batch_avg_return = np.mean( [D[i]['RET']   for i in range(batch_size)] ) #<--- will plot this at the end of training
        batch_avg_dreturn = np.mean( [D[i]['dRET']   for i in range(batch_size)] ) #<--- will plot this at the end of training

        with tt.no_grad():
            if use_q_values:
                batch_W = val(batch_S, batch_A)
            else:
                batch_W = batch_R2G - val(batch_S)
        #print(batch_S.shape, batch_A.shape, batch_W.shape)
        #raise StopIteration('Stop')
        # ------------------- ------------------- 
        # policy gradient update step (single)
        # ------------------- ------------------- 
        pie.train(True)
        opt.zero_grad()
        batch_loss = pie.log_loss(batch_S, batch_A, batch_W)
        batch_loss.backward()
        opt.step()
        lrs.step()
        pie.train(False)
    
        train_hist.append((batch_loss.item(), batch_avg_return, batch_avg_dreturn, lrs.get_last_lr()[-1])) 
        # ,  dont store loss?
        # ------------------- ------------------- 
        # policy gradient update step (single)
        # ------------------- ------------------- 
        val.train(True)
        for _ in range(viters):
            vopt.zero_grad()
            if use_q_values:
                batch_voss = voss( val(batch_S, batch_A), batch_R2G )
            else:
                batch_voss = voss( val(batch_S), batch_R2G )
            batch_voss.backward()
            vopt.step()
        vlrs.step()
        val.train(False)



        if do_validate:
            if (epoch%validation_freq==0):
                pie.switch_prediction_mode(validation_deteministic)
                if validation_episodes>1:
                    mean_validation_return, mean_validation_steps = \
                    validate_episodes(validations_envs, pie, episodes=validation_episodes,max_steps=inf,
                    validate_verbose=validation_verbose, validate_render=validation_render)
                else:
                    mean_validation_return, mean_validation_steps = \
                    validate_episode(validations_envs, pie, max_steps=inf)
                validation_hist.append((mean_validation_return, mean_validation_steps))
                print(f' [Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
                pie.switch_prediction_mode(False)
    # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
    print(f'[{100:.2f} %]')
    # validate last_time
    pie.switch_prediction_mode(validation_deteministic)
    if validation_episodes>1:
        mean_validation_return, mean_validation_steps = \
        validate_episodes(validations_envs, pie, episodes=validation_episodes,max_steps=inf,
        validate_verbose=validation_verbose, validate_render=validation_render)
    else:
        mean_validation_return, mean_validation_steps = \
        validate_episode(validations_envs, pie, max_steps=inf)
    validation_hist.append((mean_validation_return, mean_validation_steps))
    print(f' [Final-Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
    pie.switch_prediction_mode(False)

    if save_as:
        if discrete_action:
            pie.save(save_as)
        else:
            pie.save(save_as[0], save_as[1])
        print(f'Saved @ {save_as}')


    validation_hist, train_hist = np.array(validation_hist), np.array(train_hist)
    if plot_results:
        fig = plot_training_result( validation_hist, train_hist )
    else:
        fig = None

    return pie, validation_hist, train_hist, fig



def plot_training_result(validation_hist, train_hist):
        #loss, return, dreturn, lr
        tLoss, tReturn, dReturn,  tLR = \
            train_hist[:, 0], train_hist[:, 1], train_hist[:, 2], train_hist[:, 3]
        vReturn, vSteps = validation_hist[:, 0], validation_hist[:, 1]

        fig, ax = plt.subplots(2,3, figsize=(18,6))

        ax_treturn, ax_lr =    ax[0, 1], ax[1, 1]
        ax_return, ax_steps =  ax[0, 0], ax[1, 0]
        ax_dreturn, ax_loss =  ax[0, 2], ax[1, 2]

        ax_treturn.plot(tReturn, color='tab:purple', label='Avg-Return(T)')
        #ax_epsilon.scatter(np.arange(len(tEpsilon)), tEpsilon, color='tab:purple')
        ax_treturn.legend()

        ax_lr.plot(tLR, color='tab:orange', label='Learn-Rate')
        #ax_lr.scatter(np.arange(len(tLR)), tLR, color='tab:orange')
        ax_lr.legend()

        ax_return.plot(vReturn, color='tab:green', label='Return')
        ax_return.scatter(np.arange(len(vReturn)), vReturn, color='tab:green')
        ax_return.legend()

        ax_steps.plot(vSteps, color='tab:blue', label='Steps')
        ax_steps.scatter(np.arange(len(vSteps)), vSteps, color='tab:blue')
        ax_steps.legend()

        ax_dreturn.plot(dReturn, color='tab:olive', label='d-Return')
        ax_dreturn.legend()

        ax_loss.plot(tLoss, color='tab:red', label='Loss')
        ax_loss.legend()
        

        plt.show()
        return fig











#-----------------------------------------------------------------------------------------------------
""" FOOT NOTE:

def gammas(gamma, steps):
    res = np.zeros(steps, dtype='float')
    for s in range(steps):
        res[s] = gamma**(s+1)
    return res
"""
#-----------------------------------------------------------------------------------------------------