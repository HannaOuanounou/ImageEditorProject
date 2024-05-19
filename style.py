import tkinter as tk
from tkinter import ttk

def setup_window(root):
    # Définir une taille fixe pour la fenêtre
    root.geometry('800x600')  # Largeur x Hauteur
    
    # Définir une taille minimale pour la fenêtre
    root.minsize(800, 600)  # Largeur minimale, Hauteur minimale

def setup_styles():
    style = ttk.Style()
    style.configure('TButton', background='blue', foreground='grey', font=('Helvetica', 12), padding=10)
    style.configure('TFrame', background='lightgray')
    style.configure('TLabel', background='white', font=('Verdana', 10))
    style.configure('TCheckbutton', background='white', font=('Verdana', 10))

def style_button(button):
    button.config( fg='black', font=('Helvetica', 12), pady=10, padx=10)

def style_frame(frame):
    frame.config(bg='lightgray', relief=tk.RAISED, borderwidth=2)

def style_label(label):
    label.config(bg='white', fg='black', font=('Verdana', 10))

def style_checkbutton(checkbutton):
    checkbutton.config(bg='white', font=('Verdana', 10))

if __name__ == "__main__":
    root = tk.Tk()
    setup_window(root)
    setup_styles()
    
    # Créer un bouton pour tester le style
    button = ttk.Button(root, text="Test Button")
    style_button(button)
    button.pack()

    root.mainloop()
