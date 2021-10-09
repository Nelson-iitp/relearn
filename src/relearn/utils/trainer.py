import numpy as np
import matplotlib.pyplot as plt

def train(exp, pie, epochs, esteps, sample_size, report, texp, tsteps, treset, verbose=2):

    if verbose>0:
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: INFO')
        print('observation_space', exp.env.observation_space) # = gym.spaces.box.Box(-inf, inf, shape=(env.LEN,))
        print('action_space', exp.env.action_space) # = gym.spaces.discrete.Discrete(env.A)
        exp.render()
        pie.render()
        # learning loop params
        print('epochs', epochs)
        print('esteps', esteps)
        print('tsteps', tsteps)
        print('sample_size', sample_size)

        print('report', report)
        print('treset', treset)
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
            steps, reward = texp.test(pie=pie, steps=tsteps, reset_env=treset)
            hist.append([steps, reward])
            if verbose>1:
                print(' --> epoch:', epoch, ' steps:', steps, ' reward:', reward )

    # end of lerning
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TRAINER: END\tTotal Epochs:', epoch)
    

    if verbose>0:
        exp.render()
        pie.render()
        
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



def test(texp, pie, epochs, tsteps, treset=True, verbose=2):

    if verbose>0:
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: INFO')
        print('observation_space', texp.env.observation_space) # = gym.spaces.box.Box(-inf, inf, shape=(env.LEN,))
        print('action_space', texp.env.action_space) # = gym.spaces.discrete.Discrete(env.A)

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
        steps, reward = texp.test(pie=pie, steps=tsteps, reset_env=treset)
        hist.append([steps, reward])
        if verbose>1:
            print(' --> epoch:', epoch, ' steps:', steps, ' reward:', reward )

    # end of test
    print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  TESTER: END\tTotal Epochs:', epoch)
    

    if verbose>0:
        #pie.render()
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