import numpy as np
import trimesh
import itertools
import json
import pandas as pd
import os
import imageio
import matplotlib.pyplot as plt

NOMS_KEYPOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "head_top", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand"
]

def charger_keypoints_jtr(npz_path):
    data = np.load(npz_path)
    return data['Jtr']

def generer_mesures_possibles(keypoints):
    mesures = {}
    for i, j in itertools.combinations(range(len(keypoints)), 2):
        nom = f"{NOMS_KEYPOINTS[i]}_to_{NOMS_KEYPOINTS[j]}"
        distance = np.linalg.norm(keypoints[i] - keypoints[j])
        mesures[nom] = round(distance, 3)
    return mesures

def extraire_mesures_importantes(mesures):
    mots_cles = ["shoulder", "wrist", "elbow", "hip", "knee", "ankle", "head", "foot", "pelvis", "neck"]
    return {k: v for k, v in mesures.items() if any(mot in k for mot in mots_cles)}

def sauvegarder_mesures(mesures, dossier_sortie, nom_fichier="mesures_star"):
    os.makedirs(dossier_sortie, exist_ok=True)
    json_path = os.path.join(dossier_sortie, f"{nom_fichier}.json")
    csv_path = os.path.join(dossier_sortie, f"{nom_fichier}.csv")

    with open(json_path, 'w') as f:
        json.dump(mesures, f, indent=4)

    df = pd.DataFrame(mesures.items(), columns=["mesure", "valeur_m"])
    df.to_csv(csv_path, index=False)

def sauvegarder_keypoints_obj(keypoints, dossier_sortie, nom_fichier="keypoints"):
    os.makedirs(dossier_sortie, exist_ok=True)
    obj_path = os.path.join(dossier_sortie, f"{nom_fichier}.obj")
    with open(obj_path, 'w') as f:
        for pt in keypoints:
            f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
    print(f"✅ Keypoints OBJ sauvegardés dans : {obj_path}")

def interface_utilisateur_saisie(mesures_disponibles):
    print("\n🧍‍♂️ Entrez vos mesures personnelles (en mètres). Appuyez sur Entrée pour ignorer une mesure.")
    mesures_utilisateur = {}
    for mesure in sorted(mesures_disponibles.keys()):
        try:
            val = input(f"{mesure} : ")
            if val.strip():
                mesures_utilisateur[mesure] = float(val)
        except ValueError:
            print("⚠️ Entrée invalide. Mesure ignorée.")
    return mesures_utilisateur

def comparer_mesures(mesures_star, mesures_utilisateur):
    comparatif = []
    print("\n📊 Comparaison des mesures :\n")
    for mesure, val_user in mesures_utilisateur.items():
        val_star = mesures_star.get(mesure, None)
        if val_star is not None:
            diff = round(val_user - val_star, 3)
            comparatif.append((mesure, val_user, val_star, diff))
            print(f"{mesure:<35}  Utilisateur: {val_user:.3f} m | STAR: {val_star:.3f} m | Diff: {diff:+.3f} m")
        else:
            comparatif.append((mesure, val_user, None, None))
            print(f"{mesure:<35}  Utilisateur: {val_user:.3f} m | STAR: ❌ Non disponible")
    return comparatif

def sauvegarder_comparatif_csv(comparatif, dossier_sortie, nom_fichier="comparatif_utilisateur.csv"):
    os.makedirs(dossier_sortie, exist_ok=True)
    csv_path = os.path.join(dossier_sortie, nom_fichier)
    df = pd.DataFrame(comparatif, columns=["mesure", "valeur_utilisateur", "valeur_star", "difference"])
    df.to_csv(csv_path, index=False)
    print(f"✅ Comparatif utilisateur sauvegardé dans : {csv_path}")

def creer_gif_keypoints(keypoints, mesures_importantes, dossier_sortie, nom_fichier="keypoints_animation.gif"):
    os.makedirs(dossier_sortie, exist_ok=True)

    # On crée plusieurs images avec rotation
    angles = range(0, 360, 10)
    images = []

    scene = trimesh.Scene()
    for pt in keypoints:
        sph = trimesh.creation.icosphere(radius=0.01)
        sph.apply_translation(pt)
        scene.add_geometry(sph)
    for nom in mesures_importantes:
        try:
            i, j = [NOMS_KEYPOINTS.index(p) for p in nom.split("_to_")]
            line = trimesh.load_path([keypoints[i], keypoints[j]])
            scene.add_geometry(line)
        except Exception:
            continue

    for angle in angles:
        # Rotation autour de l'axe Y
        scene.set_camera(angles=(0, np.radians(angle), 0))
        png = scene.save_image(resolution=(600, 600), visible=True)
        images.append(imageio.v3.imread(png))

    gif_path = os.path.join(dossier_sortie, nom_fichier)
    imageio.mimsave(gif_path, images, duration=0.1)
    print(f"✅ GIF animé sauvegardé dans : {gif_path}")

def pipeline_traitement_star(dossier_base="star_1_1", genres=["female", "male", "neutral"]):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # dossier où est ce script

    for genre in genres:
        npz_path = os.path.join(base_dir, '..', dossier_base, genre, f"{genre}.npz")
        npz_path = os.path.normpath(npz_path)  # normaliser le chemin (important sous Windows)
        print(f"\n📌 Traitement du modèle {genre.upper()}")
        print(f"📂 Chemin absolu utilisé : {npz_path}")

        if not os.path.exists(npz_path):
            print(f"❌ Fichier non trouvé : {npz_path}")
            continue

        Jtr = charger_keypoints_jtr(npz_path)
        mesures = generer_mesures_possibles(Jtr)
        mesures_importantes = extraire_mesures_importantes(mesures)

        # Sauvegarde keypoints .obj
        dossier_sortie = os.path.join("mesures_output", genre)
        sauvegarder_keypoints_obj(Jtr, dossier_sortie)

        # Sauvegarde mesures
        sauvegarder_mesures(mesures_importantes, dossier_sortie)

        # Création GIF
        creer_gif_keypoints(Jtr, mesures_importantes, dossier_sortie)

        # Interaction utilisateur
        print(f"\n🔎 Mesures pour {genre} générées. Veux-tu comparer avec des valeurs utilisateur ? (o/n)")
        choix = input().lower()
        if choix == "o":
            mesures_utilisateur = interface_utilisateur_saisie(mesures_importantes)
            comparatif = comparer_mesures(mesures_importantes, mesures_utilisateur)
            sauvegarder_comparatif_csv(comparatif, dossier_sortie)

if __name__ == "__main__":
    pipeline_traitement_star()
