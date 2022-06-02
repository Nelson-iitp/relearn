""" DQN - Deep Q Networks 

    > DQN can be 
        ~ single DQN with no target
        ~ single DQN with target
        ~ double DQN (uses target as the double)
        ~ dueling DQN (architecture changes, updates are same) - use common.DLP
"""

from math import inf, nan
import torch as tt
import numpy as np
#import torch.optim as oo
import torch.nn as nn
import matplotlib.pyplot as plt
from ..pie import mVAL
from ..exp import make_exp, ExploreMode
from ..common import validate_episodes, validate_episode

# first need a mVAL wrapping a theta
def get_pie(value_theta, has_target, dtype, device):
    return mVAL(
        value_theta=value_theta,
        discrete_action=True,
        has_target=has_target,
        dtype=dtype, device=device)


def get_exp(env, memory_capacity, memory_seed, extra_spaces):
    return make_exp(
        env=env,
        memory_capacity=memory_capacity,
        memory_seed=memory_seed,
        **extra_spaces) #<---- note this returns exp, mem


def train(
        value_theta, 
        dtype, 
        device, 
        double, 
        tuf, 
        env, 
        gamma, 
        memory_capacity, 
        memory_seed, 
        extra_spaces, 
        optFun, 
        optArg, 
        lrsFun, 
        lrsArgs,
        min_memory, 
        epochs, 
        epsilonStart, 
        epsilonF, 
        epsilonSeed, 
        explore_size, 
        learn_times, 
        batch_size,
        verbf,
        plot_results,
        validations_envs, 
        validation_freq, 
        validation_episodes, 
        validation_verbose, 
        validation_render,
        save_as
        ):

    # setup policy
    has_target = (double or tuf>0)
    pie = get_pie(value_theta, has_target, dtype, device)

    # setup explorer and memory
    exp, mem = get_exp(env, memory_capacity, memory_seed, extra_spaces)

    # setup optimizer
    opt = optFun(pie.parameters(), **optArg)
    lrs = lrsFun(opt, **lrsArgs)

    # loss 
    lossF = nn.MSELoss() #<-- its important that DQN uses MSELoss only

    # ready training
    do_validate = ((len(validations_envs)>0) and validation_freq and validation_episodes)
    mean_validation_return, mean_validation_steps = nan, nan
    hyper_hist = []
    validation_hist = []
    learn_count, update_count = 0, 0
    exp.reset(clear_memory=False, episodic=False) #<-- initially non episodic
    # fill up memory
    len_memory = exp.memory.length()
    if len_memory < min_memory:
        exp.mode(ExploreMode.random)
        explored = exp.explore(min_memory-len_memory)
        print(f'[*] Explored Min-Memory [{explored}] Steps')

    exp.reset()
    exp.mode(ExploreMode.greedy, pie=pie, args=(epsilonStart, epsilonSeed))

    for epoch in range(epochs):
        epoch_ratio = epoch/epochs
        if (epoch%verbf==0):
            print(f'[{(100*epoch_ratio):.2f} %]')
    # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
        exp.epsilon = epsilonF(epoch_ratio)
        explored = exp.explore(explore_size)
        
        pie.train(True)
        # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
        for _ in range(learn_times):
            #  batch_size, dtype, device, discrete_action, replace=False
            pick, cS, nS, A, R, D, T = \
                exp.memory.prepare_batch(batch_size, dtype, device, discrete_action=True)
            I = tt.arange(0, pick)
            with tt.no_grad():
                target_ns = pie(nS, target=True)
                #------------------------------------------------------
                if not double:
                    updater, _ = tt.max(target_ns, dim=1)
                else:            
                    _, qmax_ind = tt.max(pie(nS, target=False), dim=1)
                    updater = target_ns[I, qmax_ind[I]]
                #------------------------------------------------------
                q_update = R + gamma * updater * (1 - D)


            pred = pie(cS, target=False)
            target = pred.detach().clone()
            target[I, A[I]] = q_update[I]
            loss =  lossF(pred, target) 
            opt.zero_grad()
            loss.backward()
            opt.step()
        # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
        pie.train(False)
        hyper_hist.append((exp.epsilon, lrs.get_last_lr()[-1]))
        lrs.step()
        learn_count+=1
        if (has_target):
            if learn_count % tuf == 0:
                pie.copy_target()
                update_count+=1


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

    validation_hist, hyper_hist = np.array(validation_hist), np.array(hyper_hist)
    if plot_results:
        fig = plot_training_result( validation_hist, hyper_hist )
    else:
        fig = None

    return pie, validation_hist, hyper_hist, fig


def plot_training_result(validation_hist, hyper_hist):
        tEpsilon, tLR = hyper_hist[:, 0], hyper_hist[:, 1]
        vReturn, vSteps = validation_hist[:, 0], validation_hist[:, 1]

        fig, ax = plt.subplots(2,2, figsize=(16,6))

        ax_epsilon, ax_lr =    ax[0, 1], ax[1, 1]
        ax_return, ax_steps =  ax[0, 0], ax[1, 0]

        ax_epsilon.plot(tEpsilon, color='tab:purple', label='Epsilon')
        #ax_epsilon.scatter(np.arange(len(tEpsilon)), tEpsilon, color='tab:purple')
        ax_epsilon.legend()

        ax_lr.plot(tLR, color='tab:red', label='Learn-Rate')
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


#-----------------------------------------------------------------------------------------------------
""" FOOT NOTE:

TODO:
    ~ implement Distributional DQN
    ~ implement Multi-Step Learning


[An Introduction to Deep Reinforcement Learning]
Vincent François-Lavet, Peter Henderson, Riashat Islam, Marc G. Bellemare and Joelle Pineau

> {QUOTE}:
The original DQN algorithm can combine the different variants. 
Experiments show that the combination of all the previously mentioned extensions to DQN
provides state-of-the-art performance on the Atari 2600 benchmarks,
both in terms of sample efficiency and final performance. 
Overall, a large majority of Atari games can be solved such that the deep RL
agents surpass the human level performance.

> {QUOTE}:
Some limitations remain with DQN-based approaches. Among others,
    ~ these types of algorithms are not well-suited to deal with large and/or continuous action spaces. 
    ~ they cannot explicitly learn stochastic policies. 

Modifications that address these limitations will be
discussed in the following Chapter 5, where we discuss policy-based
approaches. Actually, the next section will also show that value-based
and policy-based approaches can be seen as two facets of the same
model-free approach. Therefore, the limitations of discrete action spaces
and deterministic policies are only related to DQN

One can also note that value-based or policy-based approaches do
not make use of any model of the environment, which limits their sample efficiency

"""


#-----------------------------------------------------------------------------------------------------