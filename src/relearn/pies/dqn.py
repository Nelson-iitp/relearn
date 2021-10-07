import numpy as np
from chainer import optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
#import matplotlib.pyplot as plt
from chainer import serializers


class DQN:
    """ if tuf<=0 then its on-policy (single dqn) """
    def __init__(self,  acts, lr, dis, theta, tuf, base_model, mapper, seed=None):
        

        self.lr, self.dis, self.theta, self.acts = lr, dis,  theta, acts  
        self.tuf = tuf # target update frequency, if zero means its a on policy single DQN
        self.train_count=0
        self.update_count=0
        self.rand = np.random.default_rng(seed)
        self.base_model = base_model
        self.mapper = mapper
        
        self.Q = self.base_model.copy(mode='copy')
        self.T = self.base_model.copy(mode='copy') if (self.tuf>0) else self.Q
        self.optimizer = optimizers.Adam(alpha=self.lr)
        self.optimizer.setup(self.Q)

    def predict(self, state):
        qvals = self.Q(self.mapper(state)).array[0]
        qmaxs = np.where(qvals==np.max(qvals))[0]
        return qmaxs[0] if (len(qmaxs)==1) else np.random.choice(qmaxs, size=1)[0]

    def _clearQ(self):
        self.Q.copyparams(self.base_model, copy_persistent=True)
        if (self.tuf>0):
            self.T.copyparams(self.base_model, copy_persistent=True)
            
    def clear(self):
        self._clearQ()
        self.optimizer = optimizers.Adam(alpha=self.lr)
        self.optimizer.setup(self.Q)
        self.train_count=0
        self.update_count=0

    def learn(self, batch):
        steps = len(batch)
        cS, nS, act, reward, done = [], [], [], [], []
        for cSi, nSi, acti, rewardi, donei, tagi in batch:
            cS.append(self.mapper(cSi))
            nS.append(self.mapper(nSi))
            act.append(acti)
            reward.append(rewardi)
            done.append(donei)
        cS, nS, act, reward, done = np.array(cS), np.array(nS), np.array(act), np.array(reward), np.array(done)
        
        #1
        if self.tuf>0:
            target_next = np.argmax(self.Q(nS).array,axis=1)
            target_val = self.T(nS).array
            updater=np.zeros(steps)
            for i in range(steps):
                updater[i] = target_val[i, target_next[i]]
            updated_q_values = reward + self.dis * updater * (1 - done)
        else:
            updated_q_values = reward + self.dis * np.max(self.Q(nS).array, axis=1) * (1 - done)
            
        
        pred = self.Q(cS)
        target = np.copy(pred.array) 
        for i in range(steps):
            target[i, act[i]] = updated_q_values[i]*self.theta + target[i, act[i]]*(1-self.theta)
        loss = self.acts * F.mean_squared_error(target, pred)        
        self.Q.cleargrads()
        loss.backward() 
        self.optimizer.update() 
        self.train_count+=1
        # End Iteration (steps)
        
        if (self.tuf>0):
            if self.train_count % self.tuf == 0:
                self.T.copyparams(self.Q, copy_persistent=True)
                self.update_count+=1
        return

    def render(self):
        """ returns a string representation of a dict object for printing """
        print('=-=-=-=-==-=-=-=-=\nQ-NET\n=-=-=-=-==-=-=-=-=')
        self.Q.info()
        print('Train Count:', self.train_count)
        if (self.tuf>0):
            print('Update Count:', self.update_count)
        print('=-=-=-=-==-=-=-=-=!Q-net=-=-=-=-==-=-=-=-=')
        return
        
    def save(self, path):
        print("=-=-=-=-==-=-=-=-=\n Save@",path," \n=-=-=-=-==-=-=-=-=")
        serializers.save_npz(path, self.Q)
        return
        
    def load(self, path):
        print("=-=-=-=-==-=-=-=-=\n Load@",path," \n=-=-=-=-==-=-=-=-=")
        serializers.load_npz(path, self.base_model)
        self._clearQ()
        return