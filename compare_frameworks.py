  
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
# A Basic unit test comparing the output of all frameworks.
# Code Developed by:
# Ahmed A. A. Osman

from star.ch.star import STAR
import numpy as np
batch_size = 1

np_trans = np.random.normal(0,2,(1,3))
np_pose  = np.random.normal(0,2,(1,72))
np_betas = np.random.normal(0,2,(1,10))
model = STAR(gender='female',num_betas=10)

model.trans[:] = np_trans[0]
model.pose[:]  = np_pose[0]
model.betas[:] = np_betas[0]

from star.tf.star import STAR
import tensorflow as tf
import numpy as np

gender = 'female'
star = STAR(gender='female',num_betas=10)
trans = tf.constant(np_trans,dtype=tf.float32)
pose = tf.constant(np_pose,dtype=tf.float32)
betas = tf.constant(np_betas,dtype=tf.float32)
tf_star = star(pose,betas,trans)
print(np.sqrt(np.sum((tf_star.numpy()-model.r)**2.0)))

from star.pytorch.star import STAR
star = STAR(gender='female',num_betas=10)
import torch
import numpy as np
from torch.autograd import Variable
batch_size=1
poses = torch.cuda.FloatTensor(np_pose)
poses = Variable(poses,requires_grad=True)
betas = torch.cuda.FloatTensor(np_betas)
betas = Variable(betas,requires_grad=True)
trans = torch.cuda.FloatTensor(np_trans)
trans = Variable(trans,requires_grad=True)
d = star(poses, betas,trans)

diff = (model.r -d.cpu().detach().numpy())**2.0
print(np.mean(np.sqrt(np.sum(diff,axis=-1))))

diff = (tf_star.numpy() -d.cpu().detach().numpy())**2.0
print(np.mean(np.sqrt(np.sum(diff,axis=-1))))
