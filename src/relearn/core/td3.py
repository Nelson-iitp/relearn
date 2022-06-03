""" TD3 - Twin Delayed Deep Deterministic Policy Gradients

    ~ implements td3 {https://spinningup.openai.com/en/latest/algorithms/td3.html}
    ~ NOTE:
        TD3 is Same as DDPG, expcepts that 
            ~ it uses 2 value networks 
            ~ delayes learning policy and updating targets (policy_delay)
            ~ target smooting (policy_clip) <-- this is in addtion to noisy exploration
"""

from math import inf, nan
import torch as tt
import torch.distributions as td
import numpy as np
#import torch.optim as oo
import torch.nn as nn
import matplotlib.pyplot as plt
from ..pie import cdPIE, qVAL
from ..exp import make_exp, ExploreMode
from ..common import validate_episodes, validate_episode, clone_model

# first need a cdPIE and qVal wrapping a theta
def get_pie(policy_theta, has_target, dtype, device):
    return cdPIE(policy_theta, has_target, dtype, device)
def get_val(value_theta, has_target, dtype, device):
    return  qVAL(value_theta, False, has_target, dtype, device)


def get_exp(env, memory_capacity, memory_seed, extra_spaces):
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
        min_memory,
        extra_spaces,
        optFun, 
        optArg, 
        lrsFun, 
        lrsArgs,
        noiseF,
        noiseClip,
        polyak_val,
        polyak_pie,
        voptFun, 
        voptArg, 
        vlrsFun, 
        vlrsArgs,
        explore_size,
        batch_size,
        learn_times,
        policy_delay,
        policy_clip, #(2-tuple = (sigma, limits) )
        epochs,
        verbf,
        plot_results,
        validations_envs, 
        validation_freq, 
        validation_episodes,
        validation_verbose, 
        validation_render, 
        save_as
        ):

    # NOTE: both policy and value networks have target 
    # and target has to be update every step, tuf==1
    # but this is not usuall copy_target(), instead its polyak update

    # discrete_action = False #<-- mandatory
    # setup policy
    has_target = True 
    pie = get_pie(policy_theta, has_target, dtype, device)
    val_1 = get_val(value_theta, has_target, dtype, device)
    val_2 = get_val(clone_model(value_theta, detach=False), has_target, dtype, device)

    action_shape = env.action_space.shape
    action_low, action_high = \
        tt.tensor(env.action_space.low, dtype=dtype, device=device), \
            tt.tensor(env.action_space.high, dtype=dtype, device=device)
    policy_sigma, policy_clip_limit = policy_clip
    policy_smoother = td.Normal(
        loc=tt.zeros(size=action_shape, dtype=dtype),
        scale=tt.ones(size=action_shape, dtype=dtype) * policy_sigma
        )
    smooth_noiseF = lambda n : tt.clip( policy_smoother.sample((n,)), -policy_clip_limit, policy_clip_limit ).to(device=device)
    # setup explorer and memory
    exp, mem = get_exp(env, memory_capacity, memory_seed, extra_spaces)

    # setup optimizer
    opt = optFun(pie.parameters(), **optArg)
    lrs = lrsFun(opt, **lrsArgs)

    vopt_1 = voptFun(val_1.parameters(), **voptArg)
    vlrs_1 = vlrsFun(vopt_1, **vlrsArgs)
    vopt_2 = voptFun(val_2.parameters(), **voptArg)
    vlrs_2 = vlrsFun(vopt_2, **vlrsArgs)
    voss_1 = nn.MSELoss()
    voss_2 = nn.MSELoss()

    # ready training
    do_validate = ((len(validations_envs)>0) and validation_freq and validation_episodes)
    mean_validation_return, mean_validation_steps = nan, nan
    hyper_hist = []
    validation_hist = []
    learn_count = 0
    exp.reset(clear_memory=False, episodic=False) #<-- initially non episodic
    # fill up memory
    len_memory = exp.memory.length()
    if len_memory < min_memory:
        exp.mode(ExploreMode.random)
        explored = exp.explore(min_memory-len_memory)
        print(f'[*] Explored Min-Memory [{explored}] Steps')

    exp.reset()
    exp.mode(ExploreMode.noisy, pie=pie, args=(noiseF, noiseClip))

    for epoch in range(epochs):
        epoch_ratio = epoch/epochs
        if (epoch%verbf==0):
            print(f'[{(100*epoch_ratio):.2f} %]')
    # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
        explored = exp.explore(explore_size)
        
        pie.train(True)
        val_1.train(True)
        val_2.train(True)
        # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
        for j in range(learn_times):
            #  batch_size, dtype, device, discrete_action, replace=False
            pick, cS, nS, A, R, D, T = \
                exp.memory.prepare_batch(batch_size, dtype, device, discrete_action=False)
            #I = tt.arange(0, pick)

            with tt.no_grad(): #<=====================================
                policy_out =tt.clip( pie(nS, target=True) + smooth_noiseF(pick), 
                    min=action_low, max=action_high)
                target_val = tt.minimum(val_1(nS, policy_out, target=True), val_2(nS, policy_out, target=True))
                target = R + gamma * target_val * (1 - D)
            #<========================================================
            # note tt.clamp and tt.clip are the same, clip is an alisas for clamp

            for vopt, val, voss in ((vopt_1, val_1, voss_1), (vopt_2, val_2, voss_2)):
                vopt.zero_grad()
                pred = val(cS, A)
                #assert(pred.shape == target.shape)
                qloss =  voss(pred, target)
                qloss.backward()
                vopt.step()
            
            if j%policy_delay==0:
                opt.zero_grad()
                pred_actions = pie(cS)
                ploss = -(val(cS, pred_actions).mean())
                ploss.backward()
                opt.step()

                val.update_target(polyak_val)
                pie.update_target(polyak_pie)

        # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
        pie.train(False)
        val_1.train(False)
        val_2.train(False)
        hyper_hist.append( (lrs.get_last_lr()[-1], vlrs_1.get_last_lr()[-1])  )
        lrs.step()
        vlrs_1.step()
        vlrs_2.step()
        learn_count+=1

        if do_validate:
            if (epoch%validation_freq==0):
                if validation_episodes>1:
                    mean_validation_return, mean_validation_steps = \
                    validate_episodes(validations_envs, pie, episodes=validation_episodes,max_steps=inf,
                    validate_verbose=validation_verbose, validate_render=validation_render)
                else:
                    mean_validation_return, mean_validation_steps = \
                    validate_episode(validations_envs, pie, max_steps=inf)
                validation_hist.append((mean_validation_return, mean_validation_steps))
                print(f' [Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')
    # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
    print(f'[{100:.2f} %]')
    # validate last_time
    if validation_episodes>1:
        mean_validation_return, mean_validation_steps = \
        validate_episodes(validations_envs, pie, episodes=validation_episodes,max_steps=inf,
        validate_verbose=validation_verbose, validate_render=validation_render)
    else:
        mean_validation_return, mean_validation_steps = \
        validate_episode(validations_envs, pie, max_steps=inf)
    validation_hist.append((mean_validation_return, mean_validation_steps))
    print(f' [Final-Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}')

    if save_as:
        pie.save(save_as)
        print(f'Saved @ {save_as}')

    validation_hist, hyper_hist = \
        np.array(validation_hist), np.array(hyper_hist)
    if plot_results:
        fig = plot_training_result( validation_hist, hyper_hist )
    else:
        fig = None

    return pie, validation_hist, hyper_hist, fig


def plot_training_result(validation_hist, hyper_hist):
        tLR, tLRv = hyper_hist[:, 0], hyper_hist[:, 1]
        vReturn, vSteps = validation_hist[:, 0], validation_hist[:, 1]

        fig, ax = plt.subplots(2,2, figsize=(16,6))

        ax_lr, ax_lrv =         ax[0, 1], ax[1, 1]
        ax_return, ax_steps =   ax[0, 0], ax[1, 0]

        ax_lrv.plot(tLRv, color='tab:purple', label='LR(val)')
        #ax_epsilon.scatter(np.arange(len(tEpsilon)), tEpsilon, color='tab:purple')
        ax_lrv.legend()

        ax_lr.plot(tLR, color='tab:red', label='LR(pie)')
        #ax_lr.scatter(np.arange(len(tLR)), tLR, color='tab:orange')
        ax_lr.legend()


        ax_return.plot(vReturn, color='tab:green', label='Return')
        ax_return.scatter(np.arange(len(vReturn)), vReturn, color='tab:green')
        ax_return.legend()

        ax_steps.plot(vSteps, color='tab:blue', label='Steps')
        ax_steps.scatter(np.arange(len(vSteps)), vSteps, color='tab:blue')
        ax_steps.legend()
        
        plt.show()
        return fig








                


