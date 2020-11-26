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

import torch
import numpy as np
from pytorch.star import STAR
from torch.autograd import Variable

def get_vert_connectivity(num_verts, mesh_f):
    import scipy.sparse as sp
    vpv = sp.csc_matrix((num_verts,num_verts))
    def row(A):
        return A.reshape((1, -1))
    def col(A):
        return A.reshape((-1, 1))
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv

def get_verts_per_edge(num_verts,faces):
    import scipy.sparse as sp
    vc = sp.coo_matrix(get_vert_connectivity(num_verts, faces))
    def row(A):
        return A.reshape((1, -1))
    def col(A):
        return A.reshape((-1, 1))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]]
    return result

def edge_loss(d,smpl):
    vpe = get_verts_per_edge(6890,d.f)
    edges_for = lambda x: x[:,vpe[:,0],:] - x[:,vpe[:,1],:]
    edge_obj = edges_for(d) - edges_for(smpl)
    return edge_obj

def verts_loss(d,smpl):
    return torch.sum((d-smpl)**2.0)

def v2v_loss(d,smpl):
    return torch.mean(torch.sqrt(torch.sum((d-smpl)**2.0,axis=-1)))


def convert_smpl_2_star(smpl,MAX_ITER_EDGES,MAX_ITER_VERTS,NUM_BETAS,GENDER):
    '''
        Convert SMPL meshes to STAR
    :param smpl:
    :return:
    '''
    smpl = torch.cuda.FloatTensor(smpl)
    batch_size = smpl.shape[0]
    if batch_size > 32:
        import warnings
        warnings.warn(
            'The Default optimization parameters (MAX_ITER_EDGES,MAX_ITER_VERTS) were tested on batch size 32 or smaller batches')

    star = STAR(gender=GENDER)
    global_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    global_pose = Variable(global_pose, requires_grad=True)
    joints_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 72 - 3)))
    joints_pose = Variable(joints_pose, requires_grad=True)
    betas = torch.cuda.FloatTensor(np.zeros((batch_size, NUM_BETAS)))
    betas = Variable(betas, requires_grad=True)
    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    trans = Variable(trans, requires_grad=True)
    learning_rate = 1e-1
    optimizer = torch.optim.LBFGS([global_pose], lr=learning_rate)
    poses = torch.cat((global_pose, joints_pose), 1)
    d = star(poses, betas, trans)
    ########################################################################################################################
    # Fitting the model with an on edges objective first
    print('STAGE 1/2 - Fitting the Model on Edges Objective')
    for t in range(MAX_ITER_EDGES):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)

        def edge_loss_closure():
            loss = torch.sum(edge_loss(d, smpl) ** 2.0)
            return loss

        optimizer.zero_grad()
        edge_loss_closure().backward()
        optimizer.step(edge_loss_closure)

    optimizer = torch.optim.LBFGS([joints_pose], lr=learning_rate)
    for t in range(MAX_ITER_EDGES):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)

        def edge_loss_closure():
            loss = torch.sum(edge_loss(d, smpl) ** 2.0)
            return loss

        optimizer.zero_grad()
        edge_loss_closure().backward()
        optimizer.step(edge_loss_closure)
    ########################################################################################################################
    # Fitting the model with an on vertices objective
    print('STAGE 2/2 - Fitting the Model on a Vertex Objective')
    optimizer = torch.optim.LBFGS([joints_pose, global_pose, trans, betas], lr=learning_rate)
    for t in range(MAX_ITER_VERTS):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)

        def vertex_closure():
            loss = torch.sum(verts_loss(d, smpl) ** 2.0)
            return loss

        optimizer.zero_grad()
        vertex_closure().backward()
        optimizer.step(vertex_closure)

    ########################################################################################################################
    np_poses = poses.detach().cpu().numpy()
    np_betas = betas.detach().cpu().numpy()
    np_trans = trans.detach().cpu().numpy()
    np_star_verts = d.detach().cpu().numpy()
    ########################################################################################################################

    return np_poses, np_betas, np_trans , np_star_verts