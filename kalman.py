import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from numpy.linalg import inv
from skimage import io, transform
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(8)
torch.cuda.manual_seed(8)
np.random.seed(393398989)

true_x = np.random.rand(20,2)
true_x = np.add.accumulate(true_x,0)
noise_x = true_x + np.random.normal(0, 0.2, (20,2))
noise_x = torch.Tensor(noise_x)

class kalman_filter():
    def __init__(self, f, h, q, p, r=0.01):

	"""
	f : prediction matrix
	h : sensor matrix
	q : prediction noise matrix
	p : uncertainty convariance matrix
	r : measurement noise
	"""
	self.f = f
	self.h = h
	self.r = r
	self.q = q
	self.p = p

    def filter(self, x, measurement):

	# prediction error
	z = torch.mm(self.h, x)
	y = measurement[:,None] - z

	# kalman gain
	s = torch.mm(self.h, self.p)
	s = torch.mm(s, self.h.t()) + self.r

	k = torch.mm(self.p, self.h.t())
	k = torch.mm(k, torch.inverse(s))

	# new prediction
	x = torch.mm(k, y) + x
	x = torch.mm(self.f, x)

	# new uncertainty
	p_hat = torch.eye(x.size(0)) - torch.mm(k,self.h)
	p_hat = torch.mm(p_hat, self.p)
	self.p = torch.mm(p_hat, self.f.t())
	self.p = torch.mm(self.f, self.p) + self.q

	return x

    def forward(self, x):

	pred = torch.zeros(self.f.size(1),1) # initial prediction
	res = []

	for i in range(x.size(0)):
	  pred = self.filter(pred, x[i])
	  res.append(pred)

	return torch.cat(res,1).t()

f = torch.Tensor(
	[[1, 0, 1, 0, 0.5,   0],
	 [0, 1, 0, 1,   0, 0.5],
	 [0, 0, 1, 0,   1,   0],
	 [0, 0, 0, 1,   0,   1],
	 [0, 0, 0, 0,   1,   0],
	 [0, 0, 0, 0,   0,   1],]
	)
h = torch.Tensor([[1,0,0,0,0,0],[0,1,0,0,0,0]])
q = torch.eye(6)
p = torch.eye(6)*1000

kf = kalman_filter(f, h, q, p)

nx = kf.forward(noise_x)
print(nx)

plt.plot(true_x[:,0],true_x[:,1])
plt.plot(noise_x[:,0].numpy(),noise_x[:,1].numpy())
plt.plot(nx[:,0].numpy(),nx[:,1].numpy())
plt.legend(['true_x', 'noise_x', 'kalman x'])
plt.show()



