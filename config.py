import os 
path_star = '/ps/scratch/aosman/STAR/eccv2020_release/star/' 
data_type = 'float32'
device    = 'gpu'

if not os.path.exists(path_star):
    raise RuntimeError('Path to the STAR model does not exist!')

if data_type not in ['float16','float32','float64']:
    raise RuntimeError('Invalid data type %s'%(data_type))

if device not in ['cpu','gpu']:
    raise RuntimeError('Invalid device type')

class meta(object):
    pass 

cfg = meta()
cfg.data_type = data_type 
cfg.device = device 
cfg.path_star = path_star 