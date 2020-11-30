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
import numpy as np
import os
from losses import convert_smplx_2_star
from star.pytorch.star import STAR

########################################################################################################################
path_smplx_meshes = 'samples/smplx_meshes.npy'      #Path SMP-XL Meshes, a numpy array of SMPL verticies (batch_size x 6890 x 3)
path_save_star_parms = 'smplx2_star_meshes.npy' #Path to save the star paramters
star_gender = 'male'   #STAR Model Gender (options: male,female,neutral).
MAX_ITER_EDGES = 100   #Number of LBFGS iterations for an on edges objective
MAX_ITER_VERTS = 500   #Number of LBFGS iterations for an on vertices objective
NUM_BETAS = 20
########################################################################################################################


if not os.path.exists(path_smplx_meshes):
    raise RuntimeError('Path to Meshes does not exist! %s'%(path_smplx_meshes))

opt_parms = {'MAX_ITER_EDGES':MAX_ITER_EDGES ,
             'MAX_ITER_VERTS':MAX_ITER_VERTS,
             'NUM_BETAS':NUM_BETAS,
             'GENDER':star_gender}

print('Loading the SMPL-X Meshes...')
smplx = np.load(path_smplx_meshes)
def_transfer = np.load('def_transfer_smplx.npy', allow_pickle=True, encoding='latin1')[()]['mtx']
*smplx, = smplx
smplx = np.stack([def_transfer.dot(np.vstack((smplx_i,np.zeros_like(smplx_i)))) for smplx_i in smplx])

np_poses , np_betas , np_trans , star_verts , star_f = convert_smplx_2_star(smplx,**opt_parms)

results = {'poses':np_poses,'betas':np_betas,'trans':np_trans,'star_verts':star_verts}
print('Saving the results %s.'%(path_save_star_parms))
np.save(path_save_star_parms,results)
