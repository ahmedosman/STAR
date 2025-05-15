import os
import json
from pathlib import Path
import numpy as np
import trimesh
from collections import defaultdict

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
mesh_folders = {
    'male': 'Male_Mesh',
    'female': 'Female_Mesh',
    'neutral': 'Neutral_Mesh'
}
NUM_KEYPOINTS = 100  # nombre de sommets à générer (ajustable)
STRUCTURE_FILE = BASE_DIR / "body_measurements_structure.json"
OUTPUT_DIR = BASE_DIR / "keypoints_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# === UTILS ===
def farthest_point_sampling(points, num_samples):
    selected = [points[np.random.randint(len(points))]]
    for _ in range(1, num_samples):
        dists = np.linalg.norm(points - selected[-1], axis=1)
        idx = np.argmax(dists)
        selected.append(points[idx])
    return np.array(selected)

# === MAIN ===
def generate_keypoints(vertices, num_points=NUM_KEYPOINTS):
    return farthest_point_sampling(vertices, num_points)

def assign_indices_to_measurements(gender, measurements_dict, keypoint_indices):
    """
    Attribue arbitrairement deux indices différents pour chaque mesure.
    On utilise ici les indices des keypoints générés de manière répétable.
    """
    result = {}
    i = 0
    keys = list(measurements_dict.keys())
    for measure in keys:
        index1 = keypoint_indices[i % len(keypoint_indices)]
        index2 = keypoint_indices[(i + 1) % len(keypoint_indices)]
        result[measure] = [int(index1), int(index2)]
        i += 2
    return result

def process_all_meshes():
    # Charger la structure des mesures
    with open(STRUCTURE_FILE, 'r') as f:
        structure = json.load(f)

    # Résultat final à sauvegarder
    full_keypoints_data = defaultdict(dict)

    for gender, folder in mesh_folders.items():
        mesh_dir = BASE_DIR / folder
        if not mesh_dir.exists():
            print(f"❌ Dossier manquant : {mesh_dir}")
            continue

        for file_name in os.listdir(mesh_dir):
            if not file_name.endswith('.obj'):
                continue

            obj_path = mesh_dir / file_name
            mesh = trimesh.load(obj_path, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                continue

            vertices = mesh.vertices
            keypoints = generate_keypoints(vertices, NUM_KEYPOINTS)
            indices = np.random.choice(len(vertices), NUM_KEYPOINTS, replace=False)

            # Assignation des points aux mesures
            measurements = structure[gender]
            linear_indices = assign_indices_to_measurements(gender, measurements["linear"], indices)
            circum_indices = assign_indices_to_measurements(gender, measurements["circumference"], indices)

            # Stocker les résultats
            full_keypoints_data[gender][file_name] = {
                "linear": linear_indices,
                "circumference": circum_indices
            }

            # Optionnel : sauvegarder les keypoints dans un fichier à part
            json_path = OUTPUT_DIR / f"{gender}_{file_name.replace('.obj', '_keypoints.json')}"
            with open(json_path, 'w') as f_out:
                json.dump({
                    "linear": linear_indices,
                    "circumference": circum_indices
                }, f_out, indent=4)
                print(f"✅ Keypoints générés pour {file_name} → {json_path.name}")

    # Sauvegarde globale (par sécurité)
    with open(OUTPUT_DIR / "all_keypoints_by_gender.json", 'w') as f:
        json.dump(full_keypoints_data, f, indent=4)
        print(f"\n💾 Tous les keypoints ont été sauvegardés dans all_keypoints_by_gender.json")

# === EXECUTION ===
if __name__ == "__main__":
    process_all_meshes()
