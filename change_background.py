from rembg import remove
from PIL import Image

def change_background ( action,output_image_path):
    input_image_path=action[0]
    print(input_image_path)
    background_image_path=action[1]
    print(background_image_path)
    # Ouvrir l'image d'entrée et le nouvel arrière-plan
    input_image = Image.open(input_image_path)
    background_image = Image.open(background_image_path)

    # Supprimer l'arrière-plan de l'image d'entrée
    input_image_no_bg = remove(input_image)

    # Redimensionner le nouvel arrière-plan pour qu'il corresponde à la taille de l'image d'entrée
    background_image = background_image.resize(input_image_no_bg.size)

    # Convertir les images en mode RGBA pour la transparence
    input_image_no_bg = input_image_no_bg.convert("RGBA")
    background_image = background_image.convert("RGBA")

    # Créer une nouvelle image en combinant l'image d'entrée sans arrière-plan et le nouvel arrière-plan
    new_image = Image.alpha_composite(background_image, input_image_no_bg)


    print("new_image",new_image)
    if not output_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_image_path += '.png'
    # Sauvegarder l'image résultante
    new_image.save(output_image_path)

# Chemin de l'image d'entrée
#input_image_path = "/Users/hannaouanounou/Desktop/images/Reddit.com_3xglm9_cy4py5s.jpg"
# Chemin de l'image de l'arrière-plan
#background_image_path = "/Users/hannaouanounou/Desktop/images/Reddit.com_3x99uo_cy2o7k0.jpg"
# Chemin de l'image de sortie
#output_image_path = "/Users/hannaouanounou/Desktop/Tohar Cheni/ImageEditorProject/ImageEdited/output_image.png"

# Changer l'arrière-plan
#change_background(input_image_path, background_image_path, output_image_path)

# Afficher l'image résultante
    result_image = Image.open(output_image_path)
    result_image.show()
