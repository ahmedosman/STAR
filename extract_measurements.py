import os
import trimesh
import pyrender
import numpy as np
from PIL import Image
import json

# Dictionnaire de correspondance dossier <-> genre
mesh_folders = {
    'male': 'Male_Mesh',
    'female': 'Female_Mesh',
    'neutral': 'Neutral_Mesh'
}

# Dossier de sortie pour les rendus PNG
output_root = 'Rendered_Images'
os.makedirs(output_root, exist_ok=True)  # Créer le dossier de sortie si nécessaire

# Base dir pour les meshes
BASE_DIR = os.path.dirname(__file__)

# Chargement des structures de mensurations
with open('body_measurements_structure.json', 'r') as f:
    measurements_structure = json.load(f)

# Chargement des sommets associés aux mesures
with open('mesures_vers_sommets.json', 'r') as f:
    mesure_vers_sommets = json.load(f)

# Fonction pour calculer la distance linéaire entre deux points
def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Fonction pour calculer la circonférence autour d'un mesh
def calculate_circumference(mesh, points):
    distances = []
    for i in range(len(points)):
        p1 = mesh.vertices[points[i]]
        p2 = mesh.vertices[points[(i + 1) % len(points)]]  # Relier le dernier point au premier
        distances.append(calculate_distance(p1, p2))
    return sum(distances)

def check_valid_indices(mesh, points):
    for point in points:
        if point >= len(mesh.vertices):
            print(f"❌ L'indice {point} n'est pas valide dans ce mesh.")
            return False
    return True

# Parcourir les genres et les fichiers .obj
for gender, folder in mesh_folders.items():
    mesh_folder_path = os.path.join(BASE_DIR, folder)

    # Vérifie si le dossier existe
    if not os.path.exists(mesh_folder_path):
        print(f"❌ Le dossier {mesh_folder_path} n'existe pas.")
        continue

    # Parcourir tous les fichiers .obj dans le dossier
    for file_name in os.listdir(mesh_folder_path):
        if file_name.endswith('.obj'):
            obj_path = os.path.join(mesh_folder_path, file_name)

            # Charger le mesh
            mesh = trimesh.load(obj_path)

            if isinstance(mesh, trimesh.Trimesh):
                print(f"\n📦 Traitement du fichier {file_name} ({len(mesh.vertices)} vertices)")

                linear_measurements = {}
                circumference_measurements = {}

                # Mesures linéaires
                for measure in measurements_structure[gender]['linear']:
                    if measure in mesure_vers_sommets:
                        points = mesure_vers_sommets[measure]
                        if len(points) != 2:
                            print(f"⚠️ '{measure}' n'a pas exactement 2 points, ignoré.")
                            continue
                        if check_valid_indices(mesh, points):
                            linear_measurements[measure] = calculate_distance(mesh.vertices[points[0]], mesh.vertices[points[1]])
                    else:
                        print(f"⚠️ '{measure}' absent dans mesures_vers_sommets.")

                # Circonférences
                for measure in measurements_structure[gender]['circumference']:
                    if measure in mesure_vers_sommets:
                        points = mesure_vers_sommets[measure]
                        if len(points) < 3:
                            print(f"⚠️ '{measure}' a moins de 3 points, circonférence ignorée.")
                            continue
                        if check_valid_indices(mesh, points):
                            circumference_measurements[measure] = calculate_circumference(mesh, points)
                    else:
                        print(f"⚠️ '{measure}' absent dans mesures_vers_sommets.")

                # Résumé des mesures extraites
                if not linear_measurements and not circumference_measurements:
                    print(f"❌ Aucune mesure extraite pour {file_name}.")
                else:
                    print(f"📏 Mesures linéaires pour {file_name}: {linear_measurements}")
                    print(f"📐 Circonférences pour {file_name}: {circumference_measurements}")

                # Rendu 3D de l'objet
                mesh_render = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene = pyrender.Scene()
                scene.add(mesh_render)

                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                scene.add(camera, pose=np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1.0],
                    [0, 0, 0, 1]
                ]))

                light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                scene.add(light, pose=np.eye(4))

                r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
                color, _ = r.render(scene)
                r.delete()

                # Sauvegarder l'image PNG
                output_gender_dir = os.path.join(output_root, gender.capitalize())
                os.makedirs(output_gender_dir, exist_ok=True)
                output_path = os.path.join(output_gender_dir, file_name.replace('.obj', '.png'))
                Image.fromarray(color).save(output_path)
                print(f"✅ Rendu sauvegardé : {output_path}")

                # Sauvegarde des mensurations dans un fichier JSON
                output_measurements_path = os.path.join(output_gender_dir, file_name.replace('.obj', '_measurements.json'))
                with open(output_measurements_path, 'w') as f:
                    json.dump({
                        'linear': linear_measurements,
                        'circumference': circumference_measurements
                    }, f, indent=4)
                print(f"✅ Mensurations sauvegardées : {output_measurements_path}")
