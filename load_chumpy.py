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

from star.ch.star import STAR
import numpy as np

model = STAR(gender='female',num_betas=10)
## Assign random pose and shape parameters
model.pose[:] = np.random.rand(model.pose.size) * .2
model.betas[:] = np.random.rand(model.betas.size) * .03

for j in range(0,10):
    model.betas[:] = 0.0  #Each loop all PC components are set to 0.
    for i in np.linspace(-3,3,10): #Varying the jth component +/- 3 standard deviations
        model.betas[j] = i
   


