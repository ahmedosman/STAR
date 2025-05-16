import os
import numpy as np

def generer_smplh_faces_par_genre(base_dir):
    genres = ['female', 'male', 'neutral']

    for genre in genres:
        genre_dir = os.path.join(base_dir, genre)
        npz_model_path = os.path.join(genre_dir, f'{genre}.npz')
        output_path = os.path.join(genre_dir, 'smplh_faces.npy')

        if not os.path.isfile(npz_model_path):
            print(f"⚠️ Fichier introuvable : {npz_model_path}. Passage au genre suivant.")
            continue

        data = np.load(npz_model_path)
        if 'f' in data:
            faces = data['f']
        elif 'faces' in data:
            faces = data['faces']
        else:
            print(f"⚠️ Le fichier {npz_model_path} ne contient pas les faces ('f' ou 'faces'). Passage au genre suivant.")
            continue

        np.save(output_path, faces)
        print(f"✅ Fichier généré : {output_path} avec {faces.shape[0]} faces.")

if __name__ == "__main__":
    # chemin absolu vers le dossier star_1_1 (modifie si besoin)
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'star_1_1')
    generer_smplh_faces_par_genre(base_dir)
