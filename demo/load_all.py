path_model = '/ps/scratch/aosman/STAR/eccv2020_release/star/male/model.npy'
import numpy as np 

init_pose  = np.random.normal(0,1,72)
init_trans = np.zeros(3)
init_betas = np.random.normal(0,1,10)

from ch.serialization import load_model 
'''
    Remove dependency on opencv 
    Add the shape component 
'''
ch_model = load_model(path_model)
ch_model.pose[:] = init_pose 
ch_model.trans[:] = init_trans 
ch_model.betas[:10] = init_betas 

path_model = '/is/cluster/aosman/models/pruned_models/star/male/model.npy'
from tf.star import STAR 
'''
    Remove dependency on opencv 
    Add the shape component 
'''
import tensorflow as tf 
batch_size = 1
star = STAR()

pose  = tf.Variable(init_pose[np.newaxis,:],dtype=tf.float32)  
betas = tf.Variable(init_betas[np.newaxis,:],dtype=tf.float32)
trans = tf.Variable(init_trans[np.newaxis,:],dtype=tf.float32)

tf_verts = star.get_verts(pose,betas,trans)

print(np.sum((ch_model.r - tf_verts.numpy()[0])**2.0))

path_model = '/ps/scratch/aosman/STAR/eccv2020_release/star/male/model.npy'
from pytorch.star import STAR 
'''
    Remove dependency on opencv 
    Add the shape component 
'''
import tensorflow as tf 
batch_size = 1

star = STAR()
import torch
import numpy as np 
from torch.autograd import Variable

poses = torch.cuda.FloatTensor(init_pose[np.newaxis,:])
poses = Variable(poses,requires_grad=True)
betas = torch.cuda.FloatTensor(init_betas[np.newaxis,:])
betas = Variable(betas,requires_grad=True)
torch_model = star(poses, betas)
print(np.sum((ch_model.r - torch_model.cpu().detach().numpy()[0])**2.0))
