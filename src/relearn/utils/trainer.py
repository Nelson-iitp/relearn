import numpy as np
import matplotlib.pyplot as plt

def train(env, exp, pie, epochs, esteps, tsteps, sample_size, report, env_test_reset, verbose=2):

    if verbose>0:
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: INFO')
        print('observation_space', env.observation_space) # = gym.spaces.box.Box(-inf, inf, shape=(env.LEN,))
        print('action_space', env.action_space) # = gym.spaces.discrete.Discrete(env.A)
        exp.render()
        print('mem.capacity',exp.mem.capacity, 
                  'decayF', exp.decayF.__name__)
        pie.render()
        # learning loop params
        print('epochs', epochs)
        print('esteps', esteps)
        print('tsteps', tsteps)
        print('sample_size', sample_size)

        print('report', report)
        print('env_test_reset', env_test_reset)
        print('verbose', verbose)
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  !TRAINER: INFO')


    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: BEGIN')
    epoch, hist = 0, [] # history of test rewards ( and timesteps )
    while (epoch<epochs):
        epoch+=1

        # explore some no of steps ( and record experience in memory )
        exp.explore(pie=pie, steps=esteps)

        # sample experience from memory = replay
        samples = exp.mem.sample(sample_size, index_only=False)

        # learn from replay
        pie.learn(samples)

        # report 
        if epoch%report==0:
            # test updated policy
            steps, reward = exp.test(pie=pie, steps=tsteps, reset_env=env_test_reset)
            hist.append([steps, reward])
            if verbose>1:
                print(' --> epoch:', epoch, ' steps:', steps, ' reward:', reward )

    # end of lerning
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: END\tTotal Epochs:', epoch)
    pie.render()

    if verbose>0:
        # plot test reward history
        fig, ax = plt.subplots(2,1,figsize=(16,10), sharex=True)
        h = np.array(hist)
        fig.suptitle('TRAIN', fontsize=12)
        ax[1].plot(h[:,1], linewidth=0.6, label='reward', color='tab:blue')
        ax[1].set_ylabel('reward')

        ax[0].plot(h[:,0], linewidth=0.6, label='steps', color='tab:purple')
        ax[0].set_ylabel('steps')

        plt.show()
    return



def test(env, exp, pie, epochs, tsteps, verbose=2):

    if verbose>0:
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: INFO')
        print('observation_space', env.observation_space) # = gym.spaces.box.Box(-inf, inf, shape=(env.LEN,))
        print('action_space', env.action_space) # = gym.spaces.discrete.Discrete(env.A)

        pie.render()
        # testing loop params
        print('epochs', epochs)
        print('tsteps', tsteps)
        print('verbose', verbose)
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  !TESTER: INFO')

    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: BEGIN')
    epoch, hist = 0, [] # history of test rewards ( and timesteps )
    while (epoch<epochs):
        epoch+=1
        steps, reward = exp.test(pie=pie, steps=tsteps, reset_env=True)
        hist.append([steps, reward])
        if verbose>1:
            print(' --> epoch:', epoch, ' steps:', steps, ' reward:', reward )

    # end of lerning
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: END\tTotal Epochs:', epoch)
    pie.render()

    if verbose>0:
        # plot test reward history
        fig, ax = plt.subplots(2,1,figsize=(16,10), sharex=True)
        h = np.array(hist)
        fig.suptitle('TEST', fontsize=12)
        ax[1].plot(h[:,1], linewidth=0.6, label='reward', color='tab:blue')
        ax[1].set_ylabel('reward')

        ax[0].plot(h[:,0], linewidth=0.6, label='steps', color='tab:purple')
        ax[0].set_ylabel('steps')

        plt.show()
    return