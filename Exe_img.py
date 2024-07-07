import numpy as np
import matplotlib.pyplot as plt

def Conversion(file):

    binary_data = file.read()

    # Convertir les données binaires en valeurs numériques (octets)
    data = np.frombuffer(binary_data, dtype=np.uint8)

    # Déterminer la taille de l'image (par exemple, image carrée la plus proche)
    image_size = int(np.ceil(np.sqrt(len(data))))

    # Remplir la matrice pour correspondre à la taille de l'image
    padded_data = np.pad(data, (0, image_size**2 - len(data)), mode='constant')

    # Convertir les données rembourrées en matrice 2D
    matrix = padded_data.reshape((image_size, image_size))

    # Afficher l'image
    # plt.imshow(matrix, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Image convertie à partir du fichier binaire')
    # plt.show()

    return matrix

# # Utilisation de la fonction
# file_path = 'Setup.exe'  # Assurez-vous que ce chemin est correct
# image_matrix = binary_file_to_image(file_path)
