path_model = '/ps/scratch/aosman/STAR/eccv2020_release/star/male/model.npy'
from ch.serialization import load_model 
'''
    Remove dependency on opencv 
    Add the shape component 
'''
model = load_model(path_model)
