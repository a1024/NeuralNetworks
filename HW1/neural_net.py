import pandas as pd
import numpy as np

data=pd.read_csv('winequality-red.csv', sep=';').to_numpy()    #load data
np.random.shuffle(data)#random shuffle

totalsize=data.shape[0]
testsize=totalsize//2
print('%d samples: %d train, %d test'%(totalsize, totalsize-testsize, testsize))
test=data[:testsize, :]    #divide to train & test sets
train=data[testsize:, :]

'''
train-=train.mean(axis=0, keepdims=True)    #standardize
train/=train.std(axis=0, keepdims=True)
#print(train)
'''

def activation(x):
    return x*((x>0)*0.1+(x<0)*0.001)#leaky ReLU
def activation_derivative(x):
    return (x>0)*0.1+(x<0)*0.001#leaky ReLU

class NeuralNet:
    def __init__(self, nInputs, nHidden, nOutputs):
        #[12 inputs] FC 12x30 [30 hidden] FC 30 [1 output]
        self.nInputs=nInputs
        self.nHidden=nHidden
        self.nOutputs=nOutputs
        self.w1=np.random.normal(loc=0, scale=1, size=[self.nHidden, self.nInputs+1])
        self.w2=np.random.normal(loc=0, scale=1, size=[self.nOutputs, self.nHidden+1])
        self.x=np.zeros(shape=self.nInputs)
        self.net_h=np.zeros(shape=self.nHidden)
        self.h=np.zeros(shape=self.nHidden)
        self.net_yhat=np.zeros(shape=1)
        self.yhat=np.zeros(shape=1)
        self.y=np.zeros(shape=1)
        #print(self.w1.shape)
        #print(self.w2.shape)
    
    def forward(self, x, y)->np.float_:#eval data sample, returns error
        self.x=np.append(x, [1])
        self.y=y
        
        self.net_h=np.matmul(self.w1, self.x)
        
        self.net_h=np.append(self.net_h, [1])
        self.h=activation(self.net_h)
        
        self.net_y=np.matmul(self.w2, self.h)
        self.yhat=activation(self.net_y)
        
        return (self.yhat-y)**2
    
    def backward(self, rate):#update weights
        temp=rate*(self.yhat-self.y)*activation_derivative(self.net_y)
        self.w2-=temp*self.h
        
        sum=0
        for j in range(self.nHidden):
            sum+=self.w2[0][j]*activation_derivative(self.net_h[j])
        self.w1-=temp*sum*self.x


net=NeuralNet(11, 30, 1)
nIter=1000
learning_rate=0.0001
ndim=train.shape[1]
error_train0=0
error_train=0
np.seterr(all='raise')#convert RuntimeWarnings to errors
for it in range(nIter):
    try:
        error_train=0
        for sample in train:
            error=net.forward(sample[0:ndim-1], sample[ndim-1])
            error_train+=error
            net.backward(learning_rate)
        error_train/=train.shape[0]
        
        error_test=0
        for sample in test:
            error=net.forward(sample[0:ndim-1], sample[ndim-1])
            error_test+=error
        error_test/=test.shape[0]
        
        print('it %d: rate=%f, e_train=%f, e_test=%f'%(it, learning_rate, error_train, error_test))
    except:
        print('ERROR: it %d: rate=%f, e_train=%f'%(it, learning_rate, error_train))
        break
    if np.isnan(error_train) or np.isnan(error_test):
        break
    
    #if it!=0:
    #   learning_rate=(error_train0-error_train)**3
    error_train0=error_train

print('finish')
