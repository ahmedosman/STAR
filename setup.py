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
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = ["numpy", "chumpy", "opencv-python"]

setuptools.setup(
     name='STAR',  
     version='0.0.1',
     author="Ahmed A. A. Osman",
     author_email="ahmed.osman@tuebingen.mpg.de",
     install_requires=REQUIREMENTS,
     description="STAR: Sparse Trained Articulated Human Body Regressor",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ahmedosman/STAR",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: Other/Proprietary License",
         "Operating System :: OS Independent",
     ],
     

 )

