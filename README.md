# **Intelligent Image Editor**

This project is an **intelligent image editor** that combines several advanced deep learning techniques, including **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and **image generation**. With this tool, users can:

- Edit images through text-based instructions.
- Extract objects from images.
- Modify the background or apply transformations like rotation or flipping.
- Generate new images based on textual descriptions.


## **Features**

### 1. NLP-based Instruction Parsing
- Utilizes the **spaCy** model (`en_core_web_trf`) to interpret user commands and translate them into editing actions.

### 2. Object Detection and Extraction via CV
- Leverages **Mask R-CNN** for object detection and extraction from images.

### 3. Image Generation
- Uses **OpenAI’s DALL-E API** to generate new images based on user descriptions.

### 4. Image Editing
- Supports operations like rotation, color change, flipping, and background replacement.



## **Prerequisites**

To use this project, ensure the following are installed:
- **Python 3.x**
- **Pip**
- **Git**

---

## **Installation**

1. **Clone the Git repository**:  
   git clone https://github.com/HannaOuanounou/ImageEditorProject.git     
   cd ImageEditorProject

3. **Install the dependencies**:  
    pip install -r requirements.txt

4. **Set up environment variables**:  
Create a .env file at the project root with the following content:  
OPENAI_API_KEY=<your_openai_api_key>  
OUTPUT_DIR=./output  
Ensure that large model files are excluded from Git by checking .gitignore (already configured).


## **Running the Application**

After setting up the environment, start the application by running:  
python ImageEditor.py

## **Usage**

1. **Load Images**:
Upload up to 10 images into the editor.

2. **Enter Instructions**:
   Use the text input field to issue editing commands. Each image on the interface has a unique name (e.g., pic1, pic2, etc.), which allows you      to specify exactly which image you want to edit. You can provide simple or complex instructions, including multiple actions.

   Examples:

   **Single Action Instructions**:  
   "Rotate pic1 by 90 degrees".  
   "Change the background color of pic2 to blue".  
   "Extract the person from pic3".  

   **Compound Instructions**:  
   "Rotate pic1 by 90 degrees, then change the background color of pic1 to green".  
   "Extract the person from pic2 and place them on pic3".  
   "Flip pic1 horizontally and add a blue sky background to pic2"  


3. **Generate an Image**:
   Input a description like "Generate an image of a cat wearing a hat".


## **Contributions**

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
    git checkout -b feature/my-new-feature
3. Make your changes and commit them:
    git commit -m 'Add new feature'
4. Push the changes to your branch:
    git push origin feature/my-new-feature
5. Open a Pull Request.


## **Future Improvements & Work in Progress**

This project is a work in progress with several planned improvements:

**Fine-tuning Models**:  
   Fine-tuning the pre-trained spaCy and Mask R-CNN models for specific tasks or more accurate results.  
   
**Custom Image Generation (GAN)**:  
   Train a custom image generation model (GAN) to replace the DALL-E API.  
   
**Improved NLP Handling**:    
   Enhance the model’s ability to handle ambiguous and complex instructions.  
   
**Performance Optimization**:    
   Reduce image processing times and improve efficiency.  
   
**New Features**:    
   Add advanced filters, object-level editing, and more customizable backgrounds.



