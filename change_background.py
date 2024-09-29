import os
from PIL import Image, ImageColor
from rembg import remove

def change_background(action, color, output_image_path, scale_factor=0.8):
    try:
        # Debug: Print the action list to understand what it contains
        print(f"Debug: Action list received: {action}")
        
        input_image_path = action[0]  # This is pic2 (foreground)
        background_image_path = action[1] if len(action) > 1 else None  # This is pic1 (background)

        # Open the input image (foreground image)
        input_image = Image.open(input_image_path)
        
        if background_image_path:
            # Debug: Print the paths of the images
            print(f"Debug: Opening input image from path: {input_image_path}")
            print(f"Debug: Opening background image from path: {background_image_path}")
            
            # Open the background image
            background_image = Image.open(background_image_path)
            
            # Debug: Confirm that images were opened
            print("Debug: Input image and background image opened successfully.")
        else:
            # No background image provided, create a solid color background
            print(f"Debug: No background image provided. Using color {color} as the background.")
            background_color = ImageColor.getrgb(color)
            background_image = Image.new("RGBA", input_image.size, background_color)
            print(f"Debug: Created background with color: {background_color}")

        # Remove the background from the input image (foreground)
        input_image_no_bg = remove(input_image)
        
        # Debug: Confirm background removal
        print("Debug: Background removed from input image.")
        
        # Resize the foreground image to be smaller, creating a "foreground" effect
        new_foreground_size = (
            int(input_image_no_bg.width * scale_factor),
            int(input_image_no_bg.height * scale_factor)
        )
        input_image_no_bg = input_image_no_bg.resize(new_foreground_size, Image.LANCZOS)
        
        # Debug: Confirm resizing of the foreground image
        print(f"Debug: Resized foreground image to: {input_image_no_bg.size}")
        
        # Resize the background to match the input image size (if needed)
        background_image = background_image.resize(background_image.size, Image.LANCZOS)
        
        # Debug: Confirm resize operation
        print(f"Debug: Background image resized to: {background_image.size}")
        
        # Ensure both images are in RGBA mode
        input_image_no_bg = input_image_no_bg.convert("RGBA")
        background_image = background_image.convert("RGBA")
        
        # Position the resized foreground image at the center bottom of the background
        paste_position = (
            (background_image.width - input_image_no_bg.width) // 2,
            background_image.height - input_image_no_bg.height-20
        )
        
        # Create a new image by pasting the foreground onto the background
        background_image.paste(input_image_no_bg, paste_position, input_image_no_bg)
        new_image = background_image
        
        # Debug: Confirm image composition
        print("Debug: Image composition (background + resized input image) done.")

        # Save the resulting image
        output_path = os.path.join(output_image_path, "newBG.png")
        new_image.save(output_path)
        print(f"Debug: Image saved at: {output_path}, Image size: {new_image.size}")
        
        # Return the new image and the output path
        return new_image, output_path

    except FileNotFoundError as e:
        print(f"Debug: File not found error: {e}")
        return None
    except Exception as e:
        print(f"Debug: An error occurred: {e}")
        return None
