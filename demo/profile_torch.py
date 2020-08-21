from pytorch.star import STAR 
import tensorflow as tf 
import torch
import numpy as np 
from torch.autograd import Variable
list_batch_size = [2,4,8,16,32,64,128,256,512]
for batch_size in list_batch_size:
    star = STAR()
    poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
    poses = Variable(poses,requires_grad=True)
    betas = torch.cuda.FloatTensor(np.zeros((batch_size,10)))
    betas = Variable(betas,requires_grad=True)
    list_iterations = []
    for i in range(0,50):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        d = star(poses, betas)
        end.record()
        torch.cuda.synchronize()
        list_iterations.append(start.elapsed_time(end)/1000.0)
    print('Batch Size %d , Number of Iterations %f\n'%(batch_size,np.mean(list_iterations)))
