path_model = '/ps/scratch/aosman/STAR/eccv2020_release/star/male/model.npy'
from pytorch.star import STAR 
'''
    Remove dependency on opencv 
    Add the shape component 
'''
import tensorflow as tf 
batch_size = 10

star = STAR()
import torch
import numpy as np 
from torch.autograd import Variable

poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
poses = Variable(poses,requires_grad=True)
betas = torch.cuda.FloatTensor(np.zeros((batch_size,10)))
betas = Variable(betas,requires_grad=True)


d = star(poses, betas)
       