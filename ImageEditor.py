import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text, Checkbutton, IntVar, messagebox, Tk ,colorchooser, Scrollbar
from PIL import Image, ImageTk
from PIL import Image
print(Image.__version__)
import torchvision
print(torchvision.__version__)
import torch
import os
import style, EditFunctions
from NLP_Edit import NLP_Editor
from ObjectExtractor import ImageProcessor
from diffusers import DiffusionPipeline
import change_background
import openai
import requests
from dotenv import load_dotenv



# Afficher le chemin absolu du script en cours d'exécution
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Le script est exécuté dans le répertoire : {current_dir}")


# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Récupérer le chemin de sortie depuis le fichier .env
output_dir = os.getenv('OUTPUT_DIR')

# Vérifier si le chemin est défini, sinon lever une erreur ou définir une valeur par défaut
if not output_dir:
    raise ValueError("The OUTPUT_DIR environment variable is not set. Please set it in the .env file.")

# Si le chemin n'existe pas, le créer
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Output directory is set to: {output_dir}")

# Récupérer la clé API OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("The OpenAI API Key is not set. Please set it in the .env file.")
else:
    print(f"Using OpenAI API Key: {openai_api_key}")

class ImageVariationGenerator:
    def __init__(self):
        # Utiliser mps si disponible (Metal Performance Shaders)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    
   
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        style.setup_window(root)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.loaded_images = []
        self.image_count = 0
        self.current_image = None
        self.current_image_path = None
        self.current_checkbutton = None

        self.button_frame = Frame(root)
        self.button_frame.grid(row=0, column=0, sticky='ns')
        style.style_frame(self.button_frame)

        self.image_frame = Frame(root)
        self.image_frame.grid(row=0, column=1, sticky='nsew')
        style.style_frame(self.image_frame)

        self.scrollbar = Scrollbar(self.image_frame, orient='vertical')
        self.scrollbar.pack(side='right', fill='y')

        self.image_canvas = tk.Canvas(self.image_frame, yscrollcommand=self.scrollbar.set)
        self.image_canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.config(command=self.image_canvas.yview)

        self.image_container = Frame(self.image_canvas)
        self.image_canvas.create_window((0, 0), window=self.image_container, anchor='nw')
        self.image_container.bind("<Configure>", self.on_frame_configure)

        self.btn_open = Button(self.button_frame, text="Open Images", command=self.open_images)
        self.btn_open.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.btn_open)

        self.btn_save_all = Button(self.button_frame, text="Save Images", command=self.save_selected_images)
        self.btn_save_all.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.btn_save_all)
        
        self.btn_delete_selected = Button(self.button_frame, text="Delete", command=self.delete_selected_images)
        self.btn_delete_selected.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.btn_delete_selected)

        self.add_instruction_interface()
        self.add_edit_options()

    def on_frame_configure(self, event):
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

    def open_images(self):
        if len(self.loaded_images) >= 10:
            messagebox.showinfo("Limit Reached", "You can only upload up to 10 images.")
            return

        file_paths = filedialog.askopenfilenames()
        if file_paths:
            allowable_uploads = 10 - len(self.loaded_images)
            file_paths = file_paths[:allowable_uploads]

            for file_path in file_paths:
                if len(self.loaded_images) < 10:
                    self.image_count += 1
                    self.display_image(file_path)
                else:
                    break

    def display_image(self, image_path):
        image_id = f"pic{self.image_count}"
        frame = Frame(self.image_container)
        frame.pack(pady=5)
        style.style_frame(frame)

        id_label = Label(frame, text=image_id)
        id_label.pack()
        style.style_label(id_label)

        img = Image.open(image_path)
        img.thumbnail((200, 200), Image.LANCZOS)
        #img = img.resize((200, 200), Image.LANCZOS)
        photo_image = ImageTk.PhotoImage(img)

        img_label = Label(frame, image=photo_image)
        img_label.image = photo_image
        img_label.pack(side="left")
        style.style_label(img_label)

        var = IntVar()
        chk = Checkbutton(frame, variable=var, command=lambda: self.set_current_image(image_path, img_label, img, var, chk))
        chk.pack(side="left")
        style.style_checkbutton(chk)

        self.loaded_images.append((image_id, image_path, var, frame, img))

    def set_current_image(self, image_path, img_label, img, var, chk):
        if var.get() == 0:
            return  # Ignore if the checkbox is unchecked
        print("set_current_image called")  # Debug
        
        # Check if there's a previously selected checkbutton
        if self.current_checkbutton is not None and self.current_checkbutton != chk:
            self.current_checkbutton.deselect()

        # Update the current image and checkbutton
        self.current_image_path = image_path
        self.img_label = img_label
        self.current_image = img
        self.current_checkbutton = chk

    def delete_selected_images(self):
        print("delete_selected_images called") 
        remaining_images = []
        for image_id, image_path, var, frame, img in self.loaded_images:
            if var.get() == 1:
                frame.pack_forget()
                frame.destroy()
            else:
                remaining_images.append((image_id, image_path, var, frame, img))
        
        # Update image IDs after deletion
        self.loaded_images = []
        self.image_count = 0
        for i, (image_id, image_path, var, frame, img) in enumerate(remaining_images):
            self.image_count += 1
            new_image_id = f"pic{self.image_count}"
            frame.children["!label"].config(text=new_image_id)  # Update the ID label text
            self.loaded_images.append((new_image_id, image_path, var, frame, img))
        
        print("Updated image IDs: ", [img[0] for img in self.loaded_images])

    def save_selected_images(self):
        print("save_selected_images called")  
        directory = filedialog.askdirectory()
        if directory:
            for image_id, image_path, var, frame, img in self.loaded_images:
                if var.get():
                    try:
                        img.save(f"{directory}/{image_id}.png")
                        messagebox.showinfo("Success", f"{image_id} saved successfully!")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save {image_id}: {e}")

    def add_instruction_interface(self):
        self.instruction_frame = Frame(self.root)
        self.instruction_frame.grid(row=0, column=1, sticky="wes")
        style.style_frame(self.instruction_frame)

        self.instruction_text = Text(self.instruction_frame, width=60, height=1)
        self.instruction_text.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.instruction_text.bind("<KeyRelease>", self.adjust_text_height)
        style.style_label(self.instruction_text)

        self.btn_generate = Button(self.instruction_frame, text="Generate", command=self.process_instruction)
        self.btn_generate.grid(row=0, column=2, padx=10, pady=10)
        style.style_button(self.btn_generate)

    def adjust_text_height(self, event=None):
        # Get the current number of lines in the Text widget
        num_lines = int(self.instruction_text.index('end-1c').split('.')[0])
        min_height = 1
        # Adjust the height based on the number of lines
        new_height = max(min_height, num_lines)
        self.instruction_text.config(height=new_height)
    
    def process_instruction(self):
        # Get the instruction text from the Text widget
        instruction = self.instruction_text.get("1.0", tk.END).strip()
        if not instruction:
            messagebox.showinfo("Error", "Please enter an instruction!")
            return
        nlp_editor = NLP_Editor()
        actions = nlp_editor.parse_instruction(instruction)
        print("actionsImageEditor: ", actions)
        for action in actions:
            if action['action'] == "generate":
                self.generate_new_image(action)
            elif action['action'] == "extract":
                self.extract_objects(action)
            elif action['action'] == "merge":
                self.merge_objects(action['image_ids'])
            elif action['action'] == "Change Background":
                self.change_back(action['image_id'],action['color'])
            else:
                self.edit_image(action)


    def change_back(self, image_ids,color):
        img_paths=[]
        print(f"change bg of these pictures: {image_ids}")  # Debug statement
        for image_id in image_ids:
            for loaded_image in self.loaded_images:
                if loaded_image[0] == image_id:
                    img_path = loaded_image[1]
                    img_paths.append(img_path)
        new_back,output_path=change_background.change_background(img_paths,color,output_dir)
        print("new_back: ",new_back)
        self.display_generated_image(new_back,output_path)



    def extract_objects(self, action):
        print(f"Extracting objects with action: {action}")  # Debug statement
        image_id = action['image_id'][0]
        print("image_id: ", image_id)
        print("loaded_image: ", self.loaded_images)
        for loaded_image in self.loaded_images:
            if loaded_image[0] == image_id:
                img_path = loaded_image[1]
                img = loaded_image[4]
        extractor = ImageProcessor()
        objects = extractor.extract_object(img_path, action["object"], output_dir)
        print("outputpath: ", img_path)
        self.display_extracted_objects(objects)

    def merge_objects(self, image_ids):
        img_paths = []

        # Collect image paths based on image_ids
        for image_id in image_ids:
            for loaded_image in self.loaded_images:
                if loaded_image[0] == image_id:
                    img_path = loaded_image[1]
                    img_paths.append(img_path)
                    print(f"Debug: Added image path: {img_path}")  # Debug statement

        # Check if img_paths is empty before proceeding
        if not img_paths:
            messagebox.showwarning("Warning", "No valid images found to merge. Please check your image IDs.")
            return

        # Create an instance of ImageProcessor and merge the images
        extractor = ImageProcessor()
        merged_image, merged_image_path = extractor.merge_objects(img_paths, output_dir)

        # Check if the merge was successful
        if merged_image:
            print(f"Debug: Merged image created at: {merged_image_path}")  # Debug statement
            self.display_generated_image(merged_image, merged_image_path)
        else:
            print("Debug: Merging failed, no image to display.")  # Debug statement


    

    def generate_new_image(self, action):
        description = action.get('instruction', '')
        image_url = self.generate_image_with_openai(description)
        if image_url:
            image = Image.open(requests.get(image_url, stream=True).raw)
            print(f"Debug: Image successfully downloaded and opened. Image size: {image.size}")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = f"generated_{self.image_count}.png"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)
            print(f"Debug: Image saved at: {output_path}, Image size: {image.size}")
            #self.image_count += 1
            self.display_generated_image(image, output_path)


    def generate_image_with_openai(self, prompt):
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1,
                response_format="url"
            )
            return response['data'][0]['url']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate image: {e}")
            return None


    def display_extracted_objects(self, extracted_objects):
        print("extracted_obj: ", extracted_objects)

        for (object_img, path, label, bbox) in extracted_objects:
            self.image_count += 1
            image_id = f"pic{self.image_count}"
            #object_img.thumbnail((200, 200), Image.LANCZOS)
            object_img.resize((200, 200), Image.LANCZOS)
            photo_image = ImageTk.PhotoImage(object_img)

            frame = Frame(self.image_container)
            frame.pack(pady=5)

            id_label = Label(frame, text=image_id)
            id_label.pack()
            style.style_label(id_label)

            img_label = Label(frame, image=photo_image)
            img_label.image = photo_image
            img_label.pack(side="left")

            var = IntVar()
            chk = Checkbutton(frame, variable=var, command=lambda: self.set_current_image(path, img_label, object_img, var, chk))
            chk.pack(side="left")

            self.loaded_images.append((image_id, path, var, frame, object_img))

    def display_generated_image(self, img, output_path):
        self.image_count += 1
        image_id = f"pic{self.image_count}"
        #img.thumbnail((200, 200), Image.LANCZOS)
        img = img.resize((200, 200), Image.LANCZOS)
        photo_image = ImageTk.PhotoImage(img)

        frame = Frame(self.image_container)
        frame.pack(pady=5)

        id_label = Label(frame, text=image_id)
        id_label.pack()
        style.style_label(id_label)

        img_label = Label(frame, image=photo_image)
        img_label.image = photo_image
        img_label.pack(side="left")

        var = IntVar()
        chk = Checkbutton(frame, variable=var, command=lambda: self.set_current_image(output_path, img_label, img, var, chk))
        chk.pack(side="left")

        self.loaded_images.append((image_id, output_path, var, frame, img))

    def add_edit_options(self):
        self.edit_var = tk.StringVar(self.button_frame)
        self.edit_var.set("Edit Image")

        self.edit_options = ["Change Color", "Rotate", "Flip Left-Right", "Flip Up-Down"]
        self.edit_menu = tk.OptionMenu(self.button_frame, self.edit_var, *self.edit_options, command=self.apply_edit)
        self.edit_menu.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.edit_menu)

    def apply_edit(self, choice):
        print(f"apply_edit called with choice: {choice}")  # Debug
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not hasattr(self, 'current_image') or not self.current_image:
            messagebox.showinfo("Error", "No image selected!")
            return

        img = self.current_image
        img_path = self.current_image_path
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)

        if not isinstance(img, Image.Image):
            messagebox.showerror("Error", "The current image is not a valid Image object.")
            return

        print(f"apply_edit - current_image_path: {self.current_image_path}")
        print(f"apply_edit - current_image: {self.current_image}")

        if choice == "Change Color":
            color = colorchooser.askcolor(title="Choose color")[1]
            img = EditFunctions.change_color(img, color)
            self.current_image = img
        elif choice == "Rotate":
            img = EditFunctions.rotate_image(img, 90)
            self.current_image = img
        elif choice == "Flip Left-Right":
            img = EditFunctions.flip_image_lr(img)
            self.current_image = img
        elif choice == "Flip Up-Down":
            img = EditFunctions.flip_image_ud(img)
            self.current_image = img

        # Save the edited image
        try:
            img.save(output_path)
            print(f"Image saved successfully at {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")
            return

        # Display the saved and edited image
        self.display_generated_image(img, output_path)


    def edit_image(self, action):
        print("action: ", action)
        try:
        # Vérification de l'existence de l'image_id
            if not action['image_id']:
                raise ValueError(f"Image ID is missing for the action '{action['action']}'. Please specify the image on which to perform this action.")
            
            image_id = action['image_id'][0]
            print("image_id: ", image_id)
            print("loaded_image: ", self.loaded_images)
            
            for loaded_image in self.loaded_images:
                if loaded_image[0] == image_id:
                    img_path = loaded_image[1]
                    img = loaded_image[4]

                    nlp_edit = NLP_Editor()
                    print("action['action']: ", action['action'])
                    if action['action'] == "Change Color":
                        if action['color'] is None:
                            action['color'] = colorchooser.askcolor(title="Choose color")[1]
                            print("color: ", action['color'])
                        img = nlp_edit.apply_edit("Change Color", img, action)
                    elif action['action'] == "Rotate":
                        img = nlp_edit.apply_edit("Rotate", img, action)
                    elif action['action'] == "flip left-right":
                        img = nlp_edit.apply_edit("flip left-right", img, action)
                    elif action['action'] == "flip up-down":
                        img = nlp_edit.apply_edit("flip up-down", img, action)

                    # Ensure the output directory exists
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Construct the output path
                    output_path = os.path.join(output_dir, f"{image_id}_edited.png")

                    # Save the edited image
                    img.save(output_path)

                    # Update the thumbnail and display the edited image
                    self.display_generated_image(img, output_path)

            messagebox.showinfo("Success", "Image edited and displayed successfully!")

        except IndexError:
            # Gérer l'absence d'image_id
            messagebox.showerror("Error", f"No image ID provided for the action '{action['action']}'. Please specify the image to perform this action.")
        except ValueError as ve:
            # Afficher le message d'erreur personnalisé
            messagebox.showerror("Error", str(ve))
        except Exception as e:
            # Gérer toutes les autres erreurs
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Création de la fenêtre et lancement de l'application
root = Tk()
app = ImageEditorApp(root)
root.mainloop()
