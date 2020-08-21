

from tf.star import STAR 
import tensorflow as tf 
import time 
import numpy as np 

list_batch_size = [2,4,8,16,32,64,128,256,512]
for batch_size in list_batch_size:
    star = STAR()
    pose  = tf.random.normal((batch_size,72),dtype=tf.float32) 
    betas = tf.random.normal((batch_size,10),dtype=tf.float32)
    trans = tf.random.normal((batch_size,3),dtype=tf.float32)
    list_time = []
    for i in range(0,50):
        xstart = time.time()
        verts = star.get_verts(pose,betas,trans)
        list_time.append(time.time()-xstart)
    print('Batch Size %d, Duration %f'%(batch_size,np.mean(list_time[10:])))