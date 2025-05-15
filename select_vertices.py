import os
import trimesh
import numpy as np
from vedo import Mesh, Plotter
from pathlib import Path
import json

# Dictionnaire de correspondance dossier <-> genre
mesh_folders = {
    'male': 'Male_Mesh',
    'female': 'Female_Mesh',
    'neutral': 'Neutral_Mesh'
}

# Dossier de sortie pour les rendus PNG
output_root = Path('Rendered_Images')
output_root.mkdir(parents=True, exist_ok=True)  # Créer le dossier de sortie si nécessaire

# Base dir pour les meshes (le script est dans le dossier 'demo', donc ce sera relatif)
BASE_DIR = Path(__file__).parent  # Cela s'assure que nous sommes dans le bon répertoire, ici 'demo'

# Dictionnaire pour stocker les indices sélectionnés
# Structure : {genre: {individu: [indices_vertices]}}
selected_vertices = {"male": {}, "female": {}, "neutral": {}}

# Fonction pour charger les meshes
def load_meshes(gender, folder):
    mesh_folder_path = BASE_DIR / folder
    meshes = []
    for file_name in os.listdir(mesh_folder_path):
        if file_name.endswith('.obj'):
            obj_path = mesh_folder_path / file_name
            # Charger le mesh
            mesh = trimesh.load(obj_path)
            if isinstance(mesh, trimesh.Trimesh):
                meshes.append(Mesh(str(obj_path)).c("lightblue" if gender == 'male' else "pink"))
    return meshes

# Fonction pour afficher le mesh suivant
def show_next_mesh(gender, meshes, index):
    plotter = Plotter(title=f"Click on vertices to select them - {gender.capitalize()}", axes=1)
    plotter.show(meshes[index], interactive=False)
    return plotter


# Affichage et sélection des vertices
current_gender = 'female'  # Démarrer avec 'female'
current_indices = {"female": 0, "male": 0, "neutral": 0}

# Charger les meshes pour le genre actuel
female_meshes = load_meshes("female", mesh_folders['female'])
male_meshes = load_meshes("male", mesh_folders['male'])
neutral_meshes = load_meshes("neutral", mesh_folders['neutral'])

all_meshes = {"female": female_meshes, "male": male_meshes, "neutral": neutral_meshes}

# Interactivité avec vedo
plotter = None

# Fonction de sélection des vertices
def on_click(evt):
    global current_gender, current_index
    if not evt.actor:
        return
    mesh = evt.actor
    point_id = mesh.closest_point(evt.picked3d, return_point_id=True)  # Trouver le point sélectionné
    print(f"Selected vertex {point_id} on {current_gender} mesh")

    # Identifier l'individu actuel (par exemple, l'index du mesh)
    current_individual = current_indices[current_gender] - 1  # Subtract 1 to get the correct individual index

    # Vérifier si l'individu existe dans selected_vertices, sinon l'ajouter
    if current_individual not in selected_vertices[current_gender]:
        selected_vertices[current_gender][current_individual] = []

    # Vérifier si le vertex est déjà sélectionné pour l'individu actuel
    if point_id not in selected_vertices[current_gender][current_individual]:
        # Ajouter le vertex sélectionné à la liste des vertices sélectionnés
        selected_vertices[current_gender][current_individual].append(point_id)

        # Colorier le point sélectionné en rouge
        mesh.pointdata["Colors"] = np.ones_like(mesh.points)  # Réinitialiser la couleur de tous les points à blanc
        mesh.pointdata["Colors"][point_id] = np.array([1, 0, 0])  # Colorier le point sélectionné en rouge

        # Rafraîchir l'affichage du mesh avec la nouvelle couleur
        mesh.c("Colors")

        print(f"Vertices selected for {current_gender} (Individu {current_individual}): {selected_vertices[current_gender][current_individual]}")
    else:
        print(f"Vertex {point_id} is already selected for individu {current_individual}.")


# Fonction pour changer de mesh
def next_mesh():
    global current_gender, plotter

    meshes = all_meshes[current_gender]
    index = current_indices[current_gender]

    if index >= len(meshes):
        print(f"✅ Fin de la sélection pour {current_gender}.")
        if current_gender == "female":
            current_gender = "male"
        elif current_gender == "male":
            current_gender = "neutral"
        else:
            print("🎉 Tous les genres et leurs meshes ont été visualisés.")
            # Sauvegarde finale
            with open(BASE_DIR / "selected_vertices.json", "w") as f:
                json.dump(selected_vertices, f, indent=2)
            print("✅ Sélection des vertices terminée et sauvegardée dans selected_vertices.json")
            return

        index = 0
        current_indices[current_gender] = 0
        meshes = all_meshes[current_gender]

    print(f"Affichage du mesh {index + 1}/{len(meshes)} pour le genre {current_gender}.")
    
    # Réinitialiser ou créer un nouveau plotter à chaque fois
    if plotter:
        plotter.close()
    plotter = show_next_mesh(current_gender, meshes, index)

    current_indices[current_gender] += 1
    plotter.add_callback("mouse click", on_click)
    plotter.add_callback("key press", on_key_press)
    plotter.show(interactive=True)

def on_key_press(event):
    key = event.keypress  # ou event.keys si ça donne une liste de touches
    if key == "q":  # Quitter avec la touche 'q'
        print("❌ Quitter demandé par l'utilisateur.")
        if plotter:
            plotter.close()
    else:
        next_mesh()  # Toute autre touche passe au mesh suivant

# Initialisation
next_mesh()


# Sauvegarde des résultats
with open(BASE_DIR / "selected_vertices.json", "w") as f:
    json.dump(selected_vertices, f, indent=2)
print("✅ Sélection des vertices terminée et sauvegardée dans selected_vertices.json")
