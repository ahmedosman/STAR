# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman


from star.pytorch.star import STAR
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
    trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
    trans = Variable(trans,requires_grad=True)
    list_iterations = []
    for i in range(0,50):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        d = star(poses, betas, trans)
        end.record()
        torch.cuda.synchronize()
        list_iterations.append(start.elapsed_time(end)/1000.0)
    print('Batch Size %d , Number of Iterations %f\n'%(batch_size,np.mean(list_iterations)))
