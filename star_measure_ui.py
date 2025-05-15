import json
import numpy as np
import os
from vedo import Plotter, Points, Line, Mesh
import tkinter as tk
from tkinter import messagebox, filedialog

# --- Chargement données STAR (template + joints + faces) ---
def charger_modele_star(npz_path):
    data = np.load(npz_path)
    v_template = data['v_template']      # vertices (Nv,3)
    f = data['f']                        # faces (indices des triangles) (Nf,3)
    J_regressor = data['J_regressor']    # (Njoints, Nv)
    Jtr = J_regressor.dot(v_template)    # joints 3D (Njoints,3)
    return v_template, f, Jtr

# --- Chargement mapping mesures ---
def charger_mapping(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# --- Interface principale ---
class StarMesureApp:
    def __init__(self, master, base_dir, mapping):  # MODIF: on passe base_dir au lieu d'un npz_path fixe
        self.master = master
        self.mapping = mapping
        self.base_dir = base_dir  # dossier racine contenant male/, female/, neutral/
        self.selected_measure = None
        self.measure_values = {}  # stocker mesures utilisateur

        # Variable genre (homme/femme/neutre) avec valeur par défaut neutre
        self.gender_var = tk.StringVar(value='neutral')  # MODIF

        # Setup fenêtre principale
        master.title("STAR Mesures & Visualisation")
        master.geometry("1200x700")

        # Cadre droite UI Tkinter
        self.frame_ui = tk.Frame(master, width=400)
        self.frame_ui.pack(side=tk.RIGHT, fill=tk.Y)

        # Choix du genre
        tk.Label(self.frame_ui, text="Genre du modèle :", font=("Arial", 12, "bold")).pack(pady=10)
        genders = [("Homme", "male"), ("Femme", "female"), ("Neutre", "neutral")]
        for text, val in genders:
            tk.Radiobutton(self.frame_ui, text=text, variable=self.gender_var, value=val, command=self.charger_modele).pack(anchor=tk.W, padx=20)

        # Label titre
        tk.Label(self.frame_ui, text="Mesures Utilisateur", font=("Arial", 14, "bold")).pack(pady=10)

        # Liste mesures + inputs
        self.entries = {}
        for mesure, info in self.mapping.items():
            frame = tk.Frame(self.frame_ui)
            frame.pack(fill=tk.X, padx=10, pady=5)
            label = tk.Label(frame, text=f"{mesure} ({info['description']}):")
            label.pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=10)
            entry.pack(side=tk.RIGHT)
            self.entries[mesure] = entry
            label.bind("<Button-1>", lambda e, m=mesure: self.select_measure(m))

        # Boutons
        tk.Button(self.frame_ui, text="Valider Mesures", command=self.valider_mesures).pack(pady=15)
        tk.Button(self.frame_ui, text="Générer Mesh", command=self.generer_mesh).pack(pady=5)
        tk.Button(self.frame_ui, text="Exporter OBJ", command=self.exporter_obj).pack(pady=5)

        # Cadre vedo (3D)
        self.plotter = Plotter(title="Modèle STAR - Joints & Mesures", bg='white', size=(800, 700))
        self.plotter.show(interactive=False)

        # Charger le modèle initial neutre
        self.charger_modele()

    def charger_modele(self):
        gender = self.gender_var.get()
        npz_path = os.path.join(self.base_dir, gender, f"{gender}.npz")
        try:
            self.v_template, self.f, self.Jtr = charger_modele_star(npz_path)
            self.afficher_joints_et_mesures()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle pour {gender}.\n{e}")

    def afficher_joints_et_mesures(self):
        self.plotter.clear()
        # Joints gris
        pts = Points(self.Jtr, r=10, c='gray')
        self.plotter.add(pts)

        # Tracer segments mesures (en bleu)
        for mesure, info in self.mapping.items():
            joints = info["joints"]
            if any(j >= len(self.Jtr) for j in joints):
                continue
            if len(joints) == 2:
                line = Line(self.Jtr[joints[0]], self.Jtr[joints[1]], c='blue', lw=4)
                self.plotter.add(line)
            else:
                pt = Points([self.Jtr[joints[0]]], r=15, c='blue', alpha=0.5)
                self.plotter.add(pt)

        self.plotter.render()

    def select_measure(self, mesure):
        self.selected_measure = mesure
        self.plotter.clear()
        # Joints gris
        pts = Points(self.Jtr, r=10, c='gray')
        self.plotter.add(pts)

        # Tracer tous segments en bleu sauf celui sélectionné en rouge
        for m, info in self.mapping.items():
            joints = info["joints"]
            if any(j >= len(self.Jtr) for j in joints):
                continue
            c = 'red' if m == mesure else 'blue'
            if len(joints) == 2:
                line = Line(self.Jtr[joints[0]], self.Jtr[joints[1]], c=c, lw=6 if m == mesure else 3)
                self.plotter.add(line)
            else:
                pt = Points([self.Jtr[joints[0]]], r=15, c=c, alpha=0.8 if m == mesure else 0.5)
                self.plotter.add(pt)

        self.plotter.render()

    def valider_mesures(self):
        # Lecture valeurs entrées et validation
        for mesure, entry in self.entries.items():
            val_str = entry.get().strip()
            if not val_str:
                continue
            try:
                val = float(val_str)
                if val <= 0:
                    raise ValueError("Valeur négative ou nulle")
                # Stockage normalisé (en cm)
                self.measure_values[mesure] = val
            except Exception as e:
                messagebox.showerror("Erreur", f"Valeur invalide pour {mesure}: {val_str}")
                return
        messagebox.showinfo("Succès", "Mesures validées !")
        print("Mesures utilisateur :", self.measure_values)

    def generer_mesh(self):
        if not self.measure_values:
            messagebox.showwarning("Attention", "Veuillez valider vos mesures avant de générer le mesh.")
            return
        
        self.plotter.clear()
        
        # Calculer la bounding box du template
        bbox_min = self.v_template.min(axis=0)
        bbox_max = self.v_template.max(axis=0)
        bbox_size = bbox_max - bbox_min
        max_dim = bbox_size.max()
        
        # Facteur d'échelle pour redimensionner le mesh à une taille confortable (ex: 2 unités max)
        target_size = 2.0
        scale_factor = target_size / max_dim
        
        # Appliquer la mise à l'échelle sur les vertices
        v_scaled = self.v_template * scale_factor
        
        # Créer et afficher le mesh mis à l'échelle
        mesh = Mesh([v_scaled, self.f], c='lightblue', alpha=0.8)
        self.plotter.add(mesh)
        
        # Recentrer la vue automatiquement
        self.plotter.reset_camera()
        
        self.plotter.render()
        messagebox.showinfo("Info", "Mesh généré à une taille visible et confortable.")


    def exporter_obj(self):
        if self.v_template is None or len(self.v_template) == 0:
            messagebox.showerror("Erreur", "Pas de mesh à exporter.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".obj",
                                                 filetypes=[("OBJ files","*.obj")])
        if not file_path:
            return
        with open(file_path, 'w') as f:
            for v in self.v_template:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in self.f:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        messagebox.showinfo("Export", f"Mesh exporté dans {file_path}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'star_1_1')
    json_path = os.path.join(base_dir, 'star_measurements_mapping.json')

    mapping = charger_mapping(json_path)

    root = tk.Tk()
    app = StarMesureApp(root, base_dir, mapping)
    root.mainloop()
