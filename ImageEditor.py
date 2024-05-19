import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Entry, Checkbutton, IntVar, messagebox, colorchooser
import os
from PIL import Image, ImageTk
import style
import EditFunctions 
import openai
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline, DiffusionPipeline

import torch

#openai.api_key= "sk-proj-IOUKux5dOTFY3a0PHebcT3BlbkFJ46kNzU62rXgrwmr716pQ"

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        style.setup_window(root)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.loaded_images = []
        self.image_count = 0
        self.current_image = None  # Pour garder une référence à l'image courante modifiée
        self.current_image_path = None
        self.current_checkbutton = None  # Pour suivre la case à cocher actuelle

        # Frame pour les boutons à gauche
        self.button_frame = Frame(root)
        self.button_frame.grid(row=0, column=0, sticky='ns')
        style.style_frame(self.button_frame)

        # Frame pour les images au centre
        self.image_frame = Frame(root)
        self.image_frame.grid(row=0, column=1, sticky='nsew')
        style.style_frame(self.image_frame)

        # Bouton pour ouvrir des images
        self.btn_open = Button(self.button_frame, text="Open Images", command=self.open_images)
        self.btn_open.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.btn_open)

        # Bouton pour sauvegarder toutes les images
        self.btn_save_all = Button(self.button_frame, text="Save Images", command=self.save_selected_images)
        self.btn_save_all.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.btn_save_all)

        # Bouton pour supprimer les images sélectionnées
        self.btn_delete_selected = Button(self.button_frame, text="Delete", command=self.delete_selected_images)
        self.btn_delete_selected.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.btn_delete_selected)

        # Ajout de l'interface de saisie d'instructions
        self.add_instruction_interface()
        self.add_edit_options()

    def open_images(self):
        print("open_images called")  # Debug
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
        print(f"display_image called with path: {image_path}")  
        image_id = f"pic{self.image_count}"
        frame = Frame(self.image_frame)
        frame.pack(pady=5)
        style.style_frame(frame)

        id_label = Label(frame, text=image_id)
        id_label.pack()
        style.style_label(id_label)

        img = Image.open(image_path)
        img.thumbnail((200, 200), Image.LANCZOS)
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

        # Décocher la case à cocher précédente, s'il y en a une
        if self.current_checkbutton and self.current_checkbutton != chk:
            self.current_checkbutton.deselect()

        self.current_image_path = image_path
        self.img_label = img_label
        self.current_image = img
        self.current_checkbutton = chk  # Mettre à jour la case à cocher actuelle
        print(f"set_current_image - current_image_path: {self.current_image_path}")
        print(f"set_current_image - current_image: {self.current_image}")
        print(f"Current image set to {image_path}")

    def delete_selected_images(self):
        print("delete_selected_images called") 
        remaining_images = []
        for image_id, image_path, var, frame, img in self.loaded_images:
            if var.get() == 1:
                frame.pack_forget()
                frame.destroy()
            else:
                remaining_images.append((image_id, image_path, var, frame, img))
        self.loaded_images = remaining_images

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
        print("add_instruction_interface called")  
        self.instruction_frame = Frame(self.root)
        self.instruction_frame.grid(row=0, column=1, sticky="wes")
        style.style_frame(self.instruction_frame)

        self.instruction_entry = Entry(self.instruction_frame, width=60)
        self.instruction_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        style.style_label(self.instruction_entry)

        self.btn_generate = Button(self.instruction_frame, text="Generate", command=self.generate_image_from_instruction)
        self.btn_generate.grid(row=0, column=2, padx=10, pady=10)
        style.style_button(self.btn_generate)

    def generate_image_from_instruction(self):
        from accelerate import Accelerator

        # Créez un accélérateur pour gérer le calcul sur GPU
        accelerator = Accelerator()
        instruction = self.instruction_entry.get()
        if not instruction:
            messagebox.showinfo("Error", "Please enter an instruction!")
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

            pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
            pipeline.to(accelerator.device)
            image = pipeline(instruction).images[0]

            # Sauvegarder l'image générée
            output_dir = "/Users/hannaouanounou/Desktop/ImageEditProject/ImageEdited"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = f"generated_{self.image_count}.png"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)

            self.image_count += 1
            self.display_generated_image(image, output_path)
            messagebox.showinfo("Success", "Image generated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        # response = openai.Image.create(
        #     prompt=instruction,
        #     n=1,
        #     size="256x256"
        # )
        # image_url = response['data'][0]['url']
        # res = requests.get(image_url)
        # img = Image.open(BytesIO(res.content))

        # # Sauvegarder l'image générée
        # output_dir = "/Users/hannaouanounou/Desktop/ImageEditProject/ImageEdited"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # filename = f"generated_{self.image_count}.png"
        # output_path = os.path.join(output_dir, filename)
        # img.save(output_path)

        # self.image_count += 1
        # self.display_generated_image(img, output_path)
        # messagebox.showinfo("Success", "Image generated successfully!")

    def display_generated_image(self, img, output_path):
        img.thumbnail((200, 200), Image.LANCZOS)
        photo_image = ImageTk.PhotoImage(img)

        frame = Frame(self.image_frame)
        frame.pack(pady=5)
        style.style_frame(frame)

        img_label = Label(frame, image=photo_image)
        img_label.image = photo_image
        img_label.pack(side="left")
        style.style_label(img_label)

        var = IntVar()
        chk = Checkbutton(frame, variable=var, command=lambda: self.set_current_image(output_path, img_label, img, var, chk))
        chk.pack(side="left")
        style.style_checkbutton(chk)

        self.loaded_images.append((f"generated_{self.image_count}", output_path, var, frame, img))

    def add_edit_options(self):
        print("add_edit_options called")  # Debug
        self.edit_var = tk.StringVar(self.button_frame)
        self.edit_var.set("Edit Image")

        self.edit_options = ["Change Color", "Rotate", "Flip Left-Right", "Flip Up-Down"]
        self.edit_menu = tk.OptionMenu(self.button_frame, self.edit_var, *self.edit_options, command=self.apply_edit)
        self.edit_menu.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.edit_menu)


    def apply_edit(self, choice):
        print(f"apply_edit called with choice: {choice}")  # Debug
        output_dir = "/Users/hannaouanounou/Desktop/ImageEditProject/ImageEdited"
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
            if color:
                img = EditFunctions.change_color(img, output_path, color)
                self.current_image = img  # Mettre à jour l'image courante
        elif choice == "Rotate":
            img = EditFunctions.rotate_image(img, output_path, 90)
            self.current_image = img  # Mettre à jour l'image courante
        elif choice == "Flip Left-Right":
            img = EditFunctions.flip_image_lr(img, output_path)
            self.current_image = img  # Mettre à jour l'image courante
        elif choice == "Flip Up-Down":
            img = EditFunctions.flip_image_ud(img, output_path)
            self.current_image = img  # Mettre à jour l'image courante

        # Mettre à jour l'affichage de l'image
        img.thumbnail((200, 200), Image.LANCZOS)  # Assurer que l'image ne s'agrandit pas
        photo_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo_image)
        self.img_label.image = photo_image

# Création de la fenêtre et lancement de l'application
root = tk.Tk()
app = ImageEditorApp(root)
root.mainloop()
