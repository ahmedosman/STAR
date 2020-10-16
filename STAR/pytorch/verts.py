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

def verts_core():
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
    T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4,
                                                                                            4).transpose(0, 1)
    