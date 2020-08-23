from ch.star import STAR
import numpy as np
batch_size = 1
np_trans = np.random.normal(0,2,(1,3))
np_pose = np.random.normal(0,2,(1,72))
np_betas = np.random.normal(0,2,(1,10))
np_trans[:] = 0.0
np_pose[:]  = 0.0
np_betas[:] = 0.0

model = STAR(gender='female',num_betas=10)
model.trans[:] = np_trans[0]
model.pose[:]  = np_pose[0]
model.betas[:] = np_betas[0]

from tf.star import STAR
import tensorflow as tf
import numpy as np

gender = 'female'
star = STAR(gender='female',num_betas=10)
trans = tf.constant(np_trans,dtype=tf.float32)
pose = tf.constant(np_pose,dtype=tf.float32)
betas = tf.constant(np_betas,dtype=tf.float32)
tf_star = star(pose,betas,trans)
print(np.sqrt(np.sum((tf_star.numpy()-model.r)**2.0)))

from pytorch.star import STAR
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

diff = (model.v_shaped.r -d.cpu().detach().numpy())**2.0
print(np.mean(np.sqrt(np.sum(diff,axis=-1))))

diff = (tf_star.numpy() -d.cpu().detach().numpy())**2.0
print(np.mean(np.sqrt(np.sum(diff,axis=-1))))
