import os
import glob
from PIL import Image
import numpy as np
import _pickle as pickle
import random
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.01)
x += np.array([random.random()/100 for i in range(len(x))])#adds some noise
y = np.sinc(x) #makes a sinc function (non linear)
plt.plot(x,y)
plt.show()

train_input = [] #stores train x
train_output = [] #stores train y
test_input = [] #stores test x
test_output = [] #stores test y

test_indices = random.sample(list(range(len(x))), int(len(x)/4)) #chooses random 1/4th of the training dataset
test_input = x[test_indices] #stores the randomized test x
test_output = y[test_indices] #stores the randomized test y
train_input = np.delete(x,list(test_indices)) #makes train x from the leftover of test x
train_output = np.delete(y, list(test_indices)) #makes train y from the leftover of test y

#input and output shapes are [batch_size, 1]
with open('train_input.pickle', mode='wb') as f:
    pickle.dump(train_input[:,np.newaxis], f)
with open('train_output.pickle', mode='wb') as f:
    pickle.dump(train_output[:,np.newaxis], f)
with open('test_input.pickle', mode='wb') as f:
    pickle.dump(test_input[:,np.newaxis], f)
with open('test_output.pickle', mode='wb') as f:
    pickle.dump(test_output[:,np.newaxis], f)