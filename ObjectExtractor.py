import os
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms as T

class ImageProcessor:
    def __init__(self):
        self.maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.maskrcnn_model.eval()
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'backpack', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'chair',
            'apple', 'sandwich', 'donut', 'broccoli', 'banana', 'hot dog', 'pizza',
            'orange', 'cake', 'carrot', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush','0','0','0','0','book','apple','0','0'
        ]
        print("coco size: ",len(self.COCO_INSTANCE_CATEGORY_NAMES))

    def detect_objects(self, image_path, confidence_threshold=0.5):
        img = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            prediction = self.maskrcnn_model(img_tensor)

        detected_objects = []
        draw = ImageDraw.Draw(img)

        for i, (mask, box, label, score) in enumerate(zip(prediction[0]['masks'], prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores'])):
            
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = label.item()
                print("classID: ",class_id)
                label_name = self.COCO_INSTANCE_CATEGORY_NAMES[class_id]
                print(f"Detected object: {label_name} with confidence {score:.2f} at coordinates ({x1}, {y1}), ({x2}, {y2})")
                if score >= confidence_threshold:
                    object_img = np.array(img)[y1:y2, x1:x2]
                    detected_objects.append((label_name, object_img, (x1, y1, x2, y2), mask, score))
                    print(f"Added object: {label_name} with confidence {score:.2f}")

                    # Draw bounding box and label
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1), f"{label_name} {score:.2f}", fill="red")

        print("Detected objects: ", detected_objects)

        # Display the image with detections
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Detected Objects")
        plt.show()

        return detected_objects

    def extract_object(self, image_path, object_name, output_dir, confidence_threshold=0.9):
        detected_objects = self.detect_objects(image_path, confidence_threshold)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        extracted_objects = []

        for i, (label, object_img, bbox, mask, score) in enumerate(detected_objects):
            if label == object_name:
                mask_np = mask.cpu().numpy().squeeze()
                x1, y1, x2, y2 = bbox
                mask_bool = mask_np > 0.5
                extracted_region = np.zeros_like(img_np)
                extracted_region[mask_bool] = img_np[mask_bool]

                # Extraire l'image en conservant le ratio d'aspect
                original_width, original_height = x2 - x1, y2 - y1
                extracted_region_pil = Image.fromarray(extracted_region[y1:y2, x1:x2])
                
                # Optionnel: redimensionner l'objet tout en maintenant le ratio d'aspect
                target_width = 200  # Par exemple, une largeur cible fixe
                target_height = int((original_height / original_width) * target_width)
                extracted_region_pil = extracted_region_pil.resize((target_width, target_height), Image.LANCZOS)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, f"extracted_{object_name}_{i}.png")
                extracted_region_pil.save(output_path)
                extracted_objects.append((extracted_region_pil, output_path, label, bbox))

        return extracted_objects

    def plot_extracted_regions(self, image, mask, box, label):
        x1, y1, x2, y2 = box
        mask_bool = mask > 0.5
        extracted_region = np.zeros_like(image)
        extracted_region[mask_bool] = image[mask_bool]

        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(extracted_region)
        ax.set_title(f"Extracted Region - {label}")
        plt.axis('off')
        plt.show()

    
    def merge_objects(self, object_paths, output_dir):
        images = [Image.open(path) for path in object_paths]

        # Find the maximum width and height
        max_width = max(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)

        # Resize all images to have the same width or height, depending on your needs
        resized_images = []
        for img in images:
            if img.size[0] != max_width:
                new_height = int(max_width * img.size[1] / img.size[0])
                img = img.resize((max_width, new_height), Image.LANCZOS)
            elif img.size[1] != max_height:
                new_width = int(max_height * img.size[0] / img.size[1])
                img = img.resize((new_width, max_height), Image.LANCZOS)
            resized_images.append(img)

        total_width = sum(img.size[0] for img in resized_images)
        max_height = max(img.size[1] for img in resized_images)
        
        new_image = Image.new('RGBA', (total_width, max_height))

        x_offset = 0
        for img in resized_images:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width

        print("Debug: Created new merged image with size:", new_image.size)
        
        output_path = os.path.join(output_dir, f"merged_{len(object_paths)}_images.png")
        new_image.save(output_path)
        print(f"Debug: Merged image saved at: {output_path}, Image size: {new_image.size}")

        return new_image, output_path

