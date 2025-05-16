import json
import numpy as np
import matplotlib.pyplot as plt
import os

def charger_mapping(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def charger_modele_star(npz_path):
    data = np.load(npz_path)
    v_template = data['v_template']
    J_regressor = data['J_regressor']
    Jtr = J_regressor.dot(v_template)
    return v_template, Jtr

def verifier_indices_mapping(mapping):
    max_index = -1
    for mesure, info in mapping.items():
        for j in info["joints"]:
            if j > max_index:
                max_index = j
    print(f"Indice max dans mapping : {max_index}")

def afficher_mesures_3d(Jtr, mapping):
    n_joints = Jtr.shape[0]
    print(f"Nombre de joints dans le modèle: {n_joints}")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Afficher tous les joints
    ax.scatter(Jtr[:, 0], Jtr[:, 1], Jtr[:, 2], c='gray', s=40)
    for idx, (x, y, z) in enumerate(Jtr):
        ax.text(x, y, z, str(idx), color='blue', fontsize=8)

    # Pour chaque mesure, tracer la ou les lignes entre joints
    for mesure, info in mapping.items():
        joints = info["joints"]
        desc = info["description"]

        # Vérifier que les indices sont valides
        if any(j >= n_joints for j in joints):
            print(f"Attention: indices joints hors limites pour la mesure '{mesure}' : {joints}")
            continue

        # Si plusieurs joints, tracer ligne(s) entre eux
        if len(joints) == 2:
            j0, j1 = joints
            xs = [Jtr[j0, 0], Jtr[j1, 0]]
            ys = [Jtr[j0, 1], Jtr[j1, 1]]
            zs = [Jtr[j0, 2], Jtr[j1, 2]]
            ax.plot(xs, ys, zs, label=f"{mesure}: {desc}", linewidth=2)
        else:
            # Pour un seul joint, afficher un cercle (représentation simplifiée)
            j = joints[0]
            ax.scatter(Jtr[j, 0], Jtr[j, 1], Jtr[j, 2], s=100, label=f"{mesure}: {desc}", alpha=0.6)

    ax.set_title("Visualisation des mesures STAR sur joints")
    
    # Légende à l'extérieur à gauche
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.subplots_adjust(right=0.75)  # pour laisser de la place à la légende hors plot

    plt.show()

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'star_1_1')

    fichiers = {
        'female': os.path.join(base_dir, 'female', 'female.npz'),
        'male': os.path.join(base_dir, 'male', 'male.npz'),
        'neutral': os.path.join(base_dir, 'neutral', 'neutral.npz')
    }

    json_path = os.path.join(base_dir, 'star_measurements_mapping.json')

    # Choisis ici le genre du modèle à afficher
    modele_choisi = 'neutral'  # 'female' ou 'male' aussi possible

    npz_path = fichiers[modele_choisi]

    mapping = charger_mapping(json_path)
    verifier_indices_mapping(mapping)
    v_template, Jtr = charger_modele_star(npz_path)

    afficher_mesures_3d(Jtr, mapping)
