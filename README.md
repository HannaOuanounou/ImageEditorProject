Image Editor Project


This project is an intelligent image editor that combines several advanced deep learning techniques, including Natural Language Processing (NLP), Computer Vision (CV), and image generation. With this tool, users can:

-Edit images through text-based instructions.
-Extract objects from images.
-Modify the background or apply transformations like rotation or flipping.
-Generate new images based on textual descriptions.

Features
NLP-based Instruction Parsing: Uses the spaCy model (en_core_web_trf) to interpret user commands and translate them into editing actions.
Object Detection and Extraction via CV: Leverages Mask R-CNN for object detection and extraction from images.
Image Generation: Uses OpenAI’s DALL-E API to generate new images from user descriptions.
Image Editing: Supports operations like rotation, color change, flipping, and background replacement.


Prerequisites
Python 3.x
Pip
Git
Dependencies
To install the necessary dependencies, run the following command:

pip install -r requirements.txt
Clone the Git repository:

git clone https://github.com/HannaOuanounou/ImageEditorProject.git

Create a .env file at the project root and add your API keys and output directories as follows:
OPENAI_API_KEY=<your_openai_api_key>
OUTPUT_DIR=./output/
Ensure that large model files are excluded from Git by adding them to .gitignore (already configured).
Running the Application
After installing the dependencies and setting up the environment, you can start the application by running:
python ImageEditor.py


Usage
Load Images: Upload up to 10 images into the editor.
Enter Instructions: Use the text input field to issue editing commands. Examples:
"Rotate the image by 90 degrees"
"Change the background color to blue"
"Extract the person from the image"
Generate an Image: To create a new image, input a description like "Generate an image of a cat wearing a hat".

Sample Commands
Rotate: "Rotate the image by 90 degrees"
Change Color: "Change the background color to red"
Extract Objects: "Extract the car from the image"
Merge Images: "Merge pic1 and pic2"
Generate a New Image: "Generate an image of a landscape with mountains"

Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/my-new-feature).
Make your changes and commit them (git commit -m 'Add new feature').
Push the changes to your branch (git push origin feature/my-new-feature).
Open a Pull Request.
Future Improvements & Work in Progress

This project is a work in progress with several improvements planned:

Custom Image Generation (GAN): Train a custom image generation model (GAN) to replace the DALL-E API.
Improved NLP Handling: Enhance the model’s ability to handle ambiguous and complex instructions.
Performance Optimization: Focus on reducing image processing times and improving efficiency.
New Features: Add advanced filters, object-level editing, and more customizable backgrounds.


This project is licensed under the MIT License. See the LICENSE file for details.

