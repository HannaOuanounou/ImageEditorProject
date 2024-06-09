import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text, Checkbutton, IntVar, messagebox, Tk ,colorchooser
from PIL import Image, ImageTk
import torch
import os
import style,EditFunctions
from NLP_Edit import NLP_Editor  # Assurez-vous que ce module est importé correctement

class ImageVariationGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        self.pipeline.to(self.device)

    def generate_variations(self, description, n):
        images = []
        for i in range(n):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                image = self.pipeline(description).images[0]
            images.append(image)
        return images

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
        if self.current_checkbutton and self.current_checkbutton != chk:
            self.current_checkbutton.deselect()

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
        # Adjust the height of the Text widget
        self.instruction_text.config(height=num_lines)
    
    def process_instruction(self):
        # Get the instruction text from the Text widget
        instruction = self.instruction_text.get("1.0", tk.END).strip()
        if not instruction:
            messagebox.showinfo("Error", "Please enter an instruction!")
            return
        nlp_editor=NLP_Editor()
        actions = nlp_editor.parse_instruction(instruction)
        print("actionsImageEditor: ",actions)
        for action in actions:
            if action['action'] == "generate":
                self.generate_new_image(action['instruction'])
            else:
                self.edit_image(action)

    def generate_new_image(self, instruction):
        generator = ImageVariationGenerator()
        images = generator.generate_variations(instruction, 1)
        image = images[0]

        output_dir = "/Users/hannaouanounou/Desktop/ImageEditorProject/ImageEdited"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"generated_{self.image_count}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)

        self.image_count += 1
        self.display_generated_image(image, output_path)



    def display_generated_image(self, img, output_path):
        img.thumbnail((200, 200), Image.LANCZOS)
        photo_image = ImageTk.PhotoImage(img)

        frame = Frame(self.image_frame)
        frame.pack(pady=5)

        img_label = Label(frame, image=photo_image)
        img_label.image = photo_image
        img_label.pack(side="left")

        var = IntVar()
        chk = Checkbutton(frame, variable=var, command=lambda: self.set_current_image(output_path, img_label, img, var, chk))
        chk.pack(side="left")

        self.loaded_images.append((f"generated_{self.image_count}", output_path, var, frame, img))


    def add_edit_options(self):
        self.edit_var = tk.StringVar(self.button_frame)
        self.edit_var.set("Edit Image")

        self.edit_options = ["Change Color", "Rotate", "Flip Left-Right", "Flip Up-Down"]
        self.edit_menu = tk.OptionMenu(self.button_frame, self.edit_var, *self.edit_options, command=self.apply_edit)
        self.edit_menu.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        style.style_button(self.edit_menu)

    def apply_edit(self, choice):
        
        print(f"apply_edit called with choice: {choice}")  # Debug
        output_dir = "/Users/hannaouanounou/Desktop/Tohar Cheni/ImageEditorProject/ImageEdited"
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
                self.current_image = img  # Mettre à jour l'image courante
        elif choice == "Rotate":
            img = EditFunctions.rotate_image(img, 90)
            self.current_image = img  # Mettre à jour l'image courante
        elif choice == "Flip Left-Right":
            img = EditFunctions.flip_image_lr(img)
            self.current_image = img  # Mettre à jour l'image courante
        elif choice == "Flip Up-Down":
            img = EditFunctions.flip_image_ud(img)
            self.current_image = img  # Mettre à jour l'image courante

        # Mettre à jour l'affichage de l'image
        img.thumbnail((200, 200), Image.LANCZOS)  # Assurer que l'image ne s'agrandit pas
        photo_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo_image)
        self.img_label.image = photo_image

    def edit_image(self, action):
        output_path="/Users/hannaouanounou/Desktop/Tohar Cheni/ImageEditorProject/ImageEdited/"
        print("action: ",action)
        image_id = action['image_id']
        print("image_id: ", image_id)
        for loaded_image in self.loaded_images:
            if loaded_image[0] == image_id:
                img_path = loaded_image[1]
                img = loaded_image[4]

                nlp_edit = NLP_Editor()
                print("action['action']: ",action['action'])
                if action['action'] == "Change Color" :
                    if action['color']==None:
                        action['color'] = colorchooser.askcolor(title="Choose color")[1]
                        print("color: ",action['color'])
                        print("action: ",action)
                    img = nlp_edit.apply_edit("Change Color", img,action)
                    #self.current_image = img
                elif action['action'] == "Rotate" :
                    img = nlp_edit.apply_edit("Rotate", img, action)
                    #self.current_image = img
                elif action['action'] == "flip left-right":
                    img = nlp_edit.apply_edit("flip left-right", img,action)
                elif action['action'] == "flip up-down":
                    img = nlp_edit.apply_edit("flip up-down", img,action)

                if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_path += image_id+'.png'


                # Save the edited image
                img.save(output_path)
                img.thumbnail((200, 200), Image.LANCZOS)
                photo_image = ImageTk.PhotoImage(img)
                loaded_image[3].children["!label2"].config(image=photo_image)
                loaded_image[3].children["!label2"].image = photo_image

        messagebox.showinfo("Success", "Images edited successfully!")

# Création de la fenêtre et lancement de l'application
root = Tk()
app = ImageEditorApp(root)
root.mainloop()
