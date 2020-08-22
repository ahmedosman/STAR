#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2016 Max Planck Society. All rights reserved.
# Modified by Ahmed A. A. Osman Feb 2020

import chumpy
import numpy
import scipy.sparse as sp
from chumpy.ch import MatVecMult
import chumpy as ch
from .utils import verts_core , axis2quat , verts_core

def ischumpy(x):
    return hasattr(x, 'dterms')

def verts_decorated_quat(trans,
                    pose,
                    v_template,
                    J,
                    weights,
                    kintree_table,
                    f,
                    posedirs=None,
                    betas=None,
                    add_shape=True,
                    shapedirs=None,
                    want_Jtr=False):

    for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
        if which is not None:
            assert ischumpy(which)
    v = v_template

    if shapedirs is not None:
        if betas is None:
            betas = chumpy.zeros(shapedirs.shape[-1])
        if add_shape:
            v_shaped = v + shapedirs.dot(betas) #Add Shape of the model.
        else:
            v_shaped = v
    else:
        v_shaped = v
    
    quaternion_angles = axis2quat(pose.reshape((-1, 3))).reshape(-1)

    shape_feat = betas[1]
    feat = ch.concatenate([quaternion_angles,shape_feat],axis=0)
    feat = quaternion_angles

    poseblends = posedirs.dot(feat)
    v_posed = v_shaped + poseblends

    v = v_posed
    regressor = J
    J_tmpx = MatVecMult(regressor, v_shaped[:, 0])
    J_tmpy = MatVecMult(regressor, v_shaped[:, 1])
    J_tmpz = MatVecMult(regressor, v_shaped[:, 2])
    J = chumpy.vstack((J_tmpx, J_tmpy, J_tmpz)).T


    result, meta = verts_core(pose, v, J, weights, kintree_table, want_Jtr=True)
    Jtr = meta.Jtr if meta is not None else None
    tr = trans.reshape((1, 3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.poseblends = poseblends
    result.quats = quaternion_angles

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    result.poseblends = poseblends
    return result