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

from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import os 
try:
    import cPickle as pickle
except ImportError:
    import pickle
from .utils import rodrigues , quat_feat
from ..config import cfg 


class STAR(nn.Module):
    def __init__(self,gender='female',num_betas=10):
        super(STAR, self).__init__()

        if gender not in ['male','female','neutral']:
            raise RuntimeError('Invalid Gender')

        if gender == 'male':
            path_model = cfg.path_male_star
        elif gender == 'female':
            path_model = cfg.path_female_star
        else:
            path_model = cfg.path_neutral_star

        if not os.path.exists(path_model):
            raise RuntimeError('Path does not exist %s' % (path_model))
        import numpy as np

        smpl_model = np.load(path_model,allow_pickle=True)
        J_regressor = smpl_model['J_regressor']
        rows,cols = np.where(J_regressor!=0)
        vals = J_regressor[rows,cols]
        self.num_betas = num_betas

        self.register_buffer('J_regressor', torch.cuda.FloatTensor(J_regressor))
        self.register_buffer('weights', torch.cuda.FloatTensor(smpl_model['weights']))

        self.register_buffer('posedirs', torch.cuda.FloatTensor(smpl_model['posedirs'].reshape((-1,93))))
        self.register_buffer('v_template', torch.cuda.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs', torch.cuda.FloatTensor(np.array(smpl_model['shapedirs'][:,:,:num_betas])))
        self.register_buffer('faces', torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.f = smpl_model['f']

        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.verts = None
        self.J = None
        self.R = None

    def forward(self, pose, betas , trans):
        '''
            forward the model analysis
        :param pose: Pose parameters.
        :param beta: Beta parameters.
        :return:
        '''
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs  = self.shapedirs.view(-1, self.num_betas)[None, :].expand(batch_size, -1, -1)

        beta = betas[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        pose_quat = quat_feat(pose.view(-1, 3)).view(batch_size, -1)
        pose_feat = torch.cat((pose_quat[:,4:],beta[:,1]),1)

        R = rodrigues(pose.view(-1, 3)).view(batch_size, 24, 3, 3)
        R = R.view(batch_size, 24, 3, 3)

        posedirs = self.posedirs[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, pose_feat[:, :, None]).view(-1, 6890, 3)
        
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)
        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)

        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4,4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        v = v + trans[:,None,:]
        v.f = self.f
        v.v_posed = v_posed
        v.v_shaped = v_shaped
        return v
    
    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        JOINTS_IDX = [e for e in np.arange(24)]
        joints = torch.einsum("bik,ji->bjk", [vertices, self.J_regressor])
        joints = joints[:, JOINTS_IDX]
        return joints
