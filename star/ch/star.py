# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der
# Wissenschaften e.V. (MPG), acting on behalf of its Max Planck
# Institute for Intelligent Systems and the Max Planck Institute
# for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V.
# (MPG) is holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed a license
# agreement with MPG or you get the right to use the computer program
# from someone who is authorized to grant you that right. Any use of
# the computer program without a valid license is prohibited and liable
# to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing
# the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor
# <https://arxiv.org/pdf/2008.08535.pdf>
#
# Code Developed by:
# Ahmed A. A. Osman

import chumpy as ch
import numpy as np
import os
from .verts import verts_decorated_quat


def STAR(path_model='neutral.npz', num_betas=10):
    if num_betas < 2:
        raise RuntimeError('Number of betas should be at least 2')
    if not os.path.exists(path_model):
        raise RuntimeError('Path does not exist %s' % (path_model))

    model_dict = np.load(path_model, allow_pickle=True)
    trans = ch.array(np.zeros(3))
    posedirs = ch.array(model_dict['posedirs'])
    v_template = ch.array(model_dict['v_template'])

    # Regressor of the model
    J_regressor = ch.array(model_dict['J_regressor'])
    # Weights
    weights = ch.array(model_dict['weights'])
    num_joints = weights.shape[1]
    kintree_table = model_dict['kintree_table']
    f = model_dict['f']
    # Betas
    betas = ch.array(np.zeros(num_betas))
    # Shape Corrective Blend shapes
    shapedirs = ch.array(model_dict['shapedirs'][:, :, :num_betas])
    # Pose Angles
    pose = ch.array(np.zeros((num_joints*3)))
    model = verts_decorated_quat(trans=trans,
                                 pose=pose,
                                 v_template=v_template,
                                 J_regressor=J_regressor,
                                 weights=weights,
                                 kintree_table=kintree_table,
                                 f=f,
                                 posedirs=posedirs,
                                 betas=betas,
                                 shapedirs=shapedirs,
                                 want_Jtr=True)
    return model
