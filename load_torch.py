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
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'STAR')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'star')))
from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import trimesh
import torch





# star = STAR(gender='female')
# betas = np.array([
#             np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
#                       2.20098416, 0.26102114, -3.07428093, 0.55708514,
#                       -3.94442258, -2.88552087])])
# num_betas=10
# batch_size=1
# m = STAR(gender='male',num_betas=num_betas)

# # Zero pose
# poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
# betas = torch.cuda.FloatTensor(betas)

# trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
# model = star.forward(poses, betas,trans)
# shaped = model.v_shaped[-1, :, :]



# # Convertir les vertices en numpy array
# vertices = shaped.detach().cpu().numpy()  # shape: [6890, 3] (6890 vertices pour SMPL/STAR)

# # Si tu as déjà les faces (la connectivité entre les vertices)
# faces = ...  # Assure-toi d'avoir les faces sous forme de tableau (par exemple, [num_faces, 3])

# # Créer un mesh trimesh
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# # Afficher la fenêtre de visualisation
# mesh.show()


# 1. Charger le modèle STAR (ex: 'female' ou 'male')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# star = STAR(gender='female', num_betas=10, device=device).to(device)

# # 2. Générer un vecteur beta aléatoire
# num_samples = 5  # Par exemple, générer 5 corps différents
# for i in range(num_samples):
#     # Générer des betas aléatoires ~ N(0, 1)
#     betas = torch.randn(1, 10).to(device)  # 10 paramètres de forme

#     # Pose neutre (debout droit)
#     pose = torch.zeros(1, 72).to(device)

#     # Translation nulle
#     trans = torch.zeros(1, 3).to(device)

#     # 3. Obtenir les vertices
#     output = star(pose, betas, trans)
#     vertices = output[0].cpu().detach().numpy().squeeze()  # Shape: (6890, 3)
#     faces = star.faces

#     # 4. Sauvegarder en .obj pour visualiser
#     mesh = trimesh.Trimesh(vertices, faces, process=False)
#     mesh.export(f"star_sample_{i}.obj")


# Initialiser le device (CPU ou GPU)
device = torch.device('cpu')
# Nombre de samples à générer
num_samples = 5  # Par exemple, générer 5 corps différents pour chaque genre

# Les genres à générer
genders = ['female', 'male', 'neutral']
folders = {'female': 'Female_Mesh', 'male': 'Male_Mesh', 'neutral': 'Neutral_Mesh'}

# Créer les dossiers si ils n'existent pas
for gender in genders:
    if not os.path.exists(folders[gender]):
        os.makedirs(folders[gender])

# Boucle pour chaque genre
for gender in genders:
    # Créer l'objet STAR pour le genre spécifié
    star = STAR(gender=gender, num_betas=10, device=device).to(device)

    for i in range(num_samples):
        # Générer des betas aléatoires ~ N(0, 1)
        betas = torch.randn(1, 10).to(device)  # 10 paramètres de forme

        # Pose neutre (debout droit)
        pose = torch.zeros(1, 72).to(device)

        # Translation nulle
        trans = torch.zeros(1, 3).to(device)

        # Obtenir les vertices
        output = star(pose, betas, trans)
        vertices = output[0].cpu().detach().numpy().squeeze()  # Shape: (6890, 3)
        faces = star.faces

        # Sauvegarder en .obj pour visualiser
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.export(f"{folders[gender]}/star_{gender}_sample_{i}.obj")
        # mesh.export(f"star_{gender}_sample_{i}.obj")  # Nom du fichier incluant le genre

print("✅ Mesh générés avec succès.")

