import cv2
import numpy as np
from PIL import Image

def change_color(img,  color):
    """
    Change the color of the image to the specified color.
    
    :param img: PIL Image object.
    :param output_path: Path to save the edited image.
    :param color: Color to apply to the image (in hex format, e.g., '#ff0000').
    :return: PIL Image object with the new color applied.
    """
    #print ("color: ", color)
    #print("img: ",img)
    # Convert the color from hex to RGB
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Convert the PIL image to a numpy array
    np_img = np.array(img)

    # Create an array of the same shape as the image filled with the RGB color
    color_array = np.full(np_img.shape, rgb_color, dtype=np.uint8)

    # Blend the color array with the original image
    blended_img = cv2.addWeighted(np_img, 0.5, color_array, 0.5, 0)

    # Convert back to PIL image
    new_img = Image.fromarray(blended_img)
    
    return new_img


    

def rotate_image(image,  angle):
    """ Rotate the image by a given angle """
    print("angle: ", angle)
    if angle==None:
        angle=90
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image

def flip_image_lr(image):
    """ Flip the image left-right """
    print("image: ",image)
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped

def flip_image_ud(image):
    """ Flip the image up-down """
    flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
    return flipped
