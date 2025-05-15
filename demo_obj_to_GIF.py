import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# Chemins vers les dossiers des Meshs
gender_dirs = {
    'male': 'Male_Mesh',
    'female': 'Female_Mesh',
    'neutral': 'Neutral_Mesh'
}

# Fonction pour charger un fichier .obj et créer la rotation de la caméra
def render_rotation_gif(obj_path, gif_path):
    # Charger le modèle .obj avec trimesh
    mesh = trimesh.load_mesh(obj_path)

    # Charger le fichier .obj du mannequin
    mesh = trimesh.load_mesh(obj_path)

    # Appliquer une transformation pour que le mannequin soit debout sur l'axe Y
    mesh.apply_transform(np.array([[1, 0, 0, 0],  # Pas de changement pour X
                               [0, 0, -1, 0], # Rotation de 180° autour de Y 
                               [0, 1, 0, 0], # Rotation de 180° autour de Y
                               [0, 0, 0, 1]])) # Aucune translation

    # Créer la figure et l'axe
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Placer le mannequin debout et l'afficher
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, cmap='viridis', edgecolor='k')

    # Fixer les limites pour une meilleure vue
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Fonction de mise à jour pour l'animation (rotation autour de l'axe Y)
    def update(frame):
        ax.view_init(elev=20, azim=frame)  # Fixe l'élévation à 20, et fait tourner autour de l'axe Y
        return ax,

    # Créer l'animation en faisant tourner autour de l'axe Y
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=50)

    # Sauvegarder l'animation sous forme de GIF
    ani.save(gif_path, writer='imagemagick', fps=20)

    # Fermer la figure après l'animation
    plt.close(fig)

# Créer un répertoire de sortie pour les GIFs
base_gif_dir = 'Rendered_GIFS'
os.makedirs(base_gif_dir, exist_ok=True)

# Générer des GIFs pour chaque modèle dans les répertoires respectifs
for gender, gender_dir in gender_dirs.items():
    # Créer un sous-dossier pour chaque genre dans Rendered_GIFS
    gender_gif_dir = os.path.join(base_gif_dir, f'{gender.capitalize()}_GIFs')
    os.makedirs(gender_gif_dir, exist_ok=True)

    # Charger les fichiers .obj du répertoire spécifique
    for obj_file in os.listdir(gender_dir):
        if obj_file.endswith(".obj"):
            obj_path = os.path.join(gender_dir, obj_file)
            gif_path = os.path.join(gender_gif_dir, f"{os.path.splitext(obj_file)[0]}.gif")
            
            # Générer et sauvegarder le GIF
            render_rotation_gif(obj_path, gif_path)
            print(f"GIF généré pour {obj_file} dans {gif_path}")
