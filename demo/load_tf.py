path_model = '/is/cluster/aosman/models/pruned_models/star/male/model.npy'
from tf.star import STAR 
'''
    Remove dependency on opencv 
    Add the shape component 
'''
import tensorflow as tf 
batch_size = 10
star = STAR()
pose  = tf.random.normal((batch_size,72),dtype=tf.float32) 
betas = tf.random.normal((batch_size,10),dtype=tf.float32)
trans = tf.random.normal((batch_size,3),dtype=tf.float32)
verts = star.get_verts(pose,betas,trans)

