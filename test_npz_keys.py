import numpy as np

npz_path = r"C:\Users\hp\Desktop\AI_TailorApp\STAR\star_1_1\female\female.npz"

with np.load(npz_path) as data:
    print("Clés disponibles dans le .npz :", data.files)

