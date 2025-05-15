import os
import trimesh
import pyrender
import numpy as np
from PIL import Image

# Dictionnaire de correspondance dossier <-> genre
mesh_folders = {
    'male': 'Male_Mesh',
    'female': 'Female_Mesh',
    'neutral': 'Neutral_Mesh'
}

# Dossier de sortie pour les rendus PNG
output_root = 'Rendered_Images'

# Créer les dossiers de sortie si nécessaire
for gender in mesh_folders:
    output_dir = os.path.join(output_root, gender.capitalize())
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir tous les fichiers .obj dans le dossier correspondant
    for file_name in os.listdir(mesh_folders[gender]):
        if file_name.endswith('.obj'):
            obj_path = os.path.join(mesh_folders[gender], file_name)

            # Charger le mesh
            mesh = trimesh.load(obj_path)

            # S'assurer qu'on a bien un Trimesh
            if isinstance(mesh, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

                # Scène de rendu
                scene = pyrender.Scene()
                scene.add(mesh)

                # Ajouter une caméra
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                scene.add(camera, pose=np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1.0],  # Caméra reculée
                    [0, 0, 0, 1]
                ]))

                # Lumière
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                scene.add(light, pose=np.eye(4))

                # Rendu offscreen
                r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
                color, _ = r.render(scene)
                r.delete()

                # Sauvegarder l'image PNG
                output_path = os.path.join(output_root, gender.capitalize(), file_name.replace('.obj', '.png'))
                Image.fromarray(color).save(output_path)
                print(f"✅ Rendu sauvegardé : {output_path}")
