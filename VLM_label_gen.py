import os
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, VipLlavaForConditionalGeneration


PROMPT_TEMPLATE = "Human: <image>\n{}\nAssistant:"
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}
VISUAL_PROMPTS = {
    "mask": "What is the name of the object highlighted with a {} mask in the image? Your output must be only the class name of the object. Do not add any additional text.",
    "bb": "What is the name of the object inside the {} rectangle in the image? Your output must be only the class name of the object. Do not add any additional text.",
    "contour": "What is the name of the object inside the {} mask contour? Your output must be only the class name of the object. Do not add any additional text.",
    "ellipse": "What is the name of the object inside the {} ellipses? Your output must be only the class name of the object. Do not add any additional text.",
}
VISUAL_PROMPTS_DESCRIPTIONS_FROM_CLASSNAME = {
    "mask": "Given the image provided, identify and provide the definition of the {} highlighted by the {} mask.",
    "bb": "Given the image provided, identify and provide the definition of the {} inside the {} rectangle.",
    "contour": "Given the image provided, identify and provide the definition of the {} inside the {} mask contour.",
    "ellipse": "Given the image provided, identify and provide the definition of the {} inside the {} ellipses. ",
}
PROMPT_ALL_OBJS = "What is the name of all the object inside the image? Your output must be only the class name of the objects inside the image, each separated by a comma. Ensure that each noun is written in its singular form. Do not use plural forms or irregular plurals. Do not add any additional text. Example output: apple,banana,orange"


class EnsambleConfig:
    def __init__(
            self, 
            ensamble_prompts: bool = False,
            ensamble_zoom: bool = False,
            ensamble_colors: bool = False,
            prompt_types: list = ["bb", "contour"], 
            zoom_percentages: list = [0, 30, 50], 
            colors: list = ["red", "green", "blue"]
        ):
        self.ensamble_prompts = ensamble_prompts
        self.ensamble_zoom = ensamble_zoom
        self.ensamble_colors = ensamble_colors
        self.prompt_types = prompt_types
        self.zoom_percentages = zoom_percentages
        self.colors = colors
    
    def is_ensamble(self) -> bool:
        if self.ensamble_zoom or self.ensamble_colors:
            return True
        
        if self.ensamble_prompts and not self.ensamble_zoom and not self.ensamble_colors:
            print("[WARNING] Ensamble prompts is enabled but no other ensamble option is enabled. Using default prompt w/o ensamble.")
            return False
        
        return False
    
    def is_ensamble_color_only(self) -> bool:
        if self.ensamble_colors and not self.ensamble_zoom and not self.ensamble_prompts:
            return True
        
        return False
    
    def is_ensamble_zoom_only(self) -> bool:
        if self.ensamble_zoom and not self.ensamble_colors and not self.ensamble_prompts:
            return True
        
        return False
    
    def is_ensamble_prompt_color(self) -> bool:
        if self.ensamble_prompts and self.ensamble_colors and not self.ensamble_zoom:
            return True
        
        return False
    
    def is_ensamble_prompt_zoom(self) -> bool:
        if self.ensamble_prompts and self.ensamble_zoom and not self.ensamble_colors:
            return True
        
        return False
    
    def is_ensamble_color_zoom(self) -> bool:
        if self.ensamble_colors and self.ensamble_zoom and not self.ensamble_prompts:
            return True
        
        return False
    
    def is_full_ensamble(self) -> bool:
        if self.ensamble_prompts and self.ensamble_zoom and self.ensamble_colors:
            return True
        
        return False

class LabelGenerator:
    def __init__(self, args):
        self.ensamble_config = EnsambleConfig(
            ensamble_prompts=args.ensamble_prompts,
            ensamble_colors=args.ensamble_colors,
            ensamble_zoom=args.ensamble_zoom,
            colors=args.ensamble_colors_list,
            prompt_types=args.ensamble_prompts_list,
            zoom_percentages=args.ensamble_zoom_list
        )
        self.vlm_model = VipLlavaForConditionalGeneration.from_pretrained(os.path.join(args.vlm_model_path, "vip-llava-7b-hf"), device_map="auto", torch_dtype=torch.float32, load_in_4bit=True)
        self.vlm_processor = AutoProcessor.from_pretrained(os.path.join(args.vlm_model_path, "vip-llava-7b-hf"))

    def zoom_on_masked_object(self, image, mask, zoom_percent):
        """
        Create a zoomed-in view of the object identified by the mask.

        Args:
        image (np.array): Original image (3D numpy array for BGR images)
        mask (np.array): Binary mask (2D numpy array with values 0 or 1)
        zoom_percent (float): Percentage to zoom in (e.g., 150 for 150% zoom)

        Returns:
        np.array: Zoomed image with the same dimensions as the input image
        """
        if zoom_percent <= 0:
            return image
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image  # Return original image if no object is found

        # Find the bounding box that encompasses all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Calculate the center of the bounding box
        center_x, center_y = x + w // 2, y + h // 2

        # Calculate the new dimensions based on the zoom percentage
        new_w = int(w * (100 / zoom_percent))
        new_h = int(h * (100 / zoom_percent))

        # Ensure the new dimensions don't exceed the image size
        new_w = min(new_w, image.shape[1])
        new_h = min(new_h, image.shape[0])

        # Calculate the new top-left corner
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)

        # Adjust if the new bounding box exceeds image boundaries
        if new_x + new_w > image.shape[1]:
            new_x = image.shape[1] - new_w
        if new_y + new_h > image.shape[0]:
            new_y = image.shape[0] - new_h

        # Crop the region of interest
        cropped = image[new_y:new_y+new_h, new_x:new_x+new_w]

        # Resize the cropped image to match the original image dimensions
        result = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        return result
        
    def blend_image_with_colored_mask(self, X, mask, color, alpha):
        """
        Blend an image X with a binary mask and a specified color according to the rule:
        composite_X = alpha * (mask * color) + (1 - alpha) * X
        
        Args:
        X (np.array): Original image (3D numpy array for RGB images)
        mask (np.array): Binary mask (2D numpy array with values 0 or 1)
        color (tuple): RGB color tuple for the mask (e.g., (255, 0, 0) for red)
        alpha (float): Blending parameter between 0 and 1
        
        Returns:
        np.array: Composite image
        """
        # Ensure mask is binary
        mask = (mask > 0).astype(float)
        
        # Expand mask dimensions to match the image
        mask = np.expand_dims(mask, axis=-1)
        
        # Create a color mask
        color_mask = mask * np.array(color)
        
        # Create the composite image
        composite_X = alpha * color_mask + (1 - alpha) * X
        
        # Apply the blending only where the mask is non-zero
        result = np.where(mask, composite_X, X)
        
        return result.astype(np.uint8)

    def draw_bounding_boxes_on_masked_regions(self, image, mask, color=(255, 0, 0), thickness=2, alpha = 0.5):
        """
        Draw bounding boxes around regions identified by a binary mask on the original image.

        Args:
        image (np.array): Original image (3D numpy array for RGB images)
        mask (np.array): Binary mask (2D numpy array with values 0 or 1)
        color (tuple): RGB color tuple for the bounding boxes (default is red)
        thickness (int): Thickness of the bounding box lines
        alpha (float): Blending parameter between 0 and 1 for the bounding box

        Returns:
        np.array: Image with bounding boxes drawn
        """
        # Ensure mask is binary and of type uint8
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a transparent overlay
        overlay = image.copy()

        # Draw filled bounding boxes on the overlay
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)  # -1 for filled rectangle

        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return result

    def draw_contours(self, image, mask, color=(255, 0, 0), thickness=2, alpha=0.5):
        """
        Draw semi-transparent contours around objects identified by a binary mask on the original image.

        Args:
        image (np.array): Original image (3D numpy array for BGR images)
        mask (np.array): Binary mask (2D numpy array with values 0 or 1)
        color (tuple): BGR color tuple for the contours (default is green)
        thickness (int): Thickness of the contour lines
        alpha (float): Transparency of the contours (0.0 to 1.0)

        Returns:
        np.array: Image with semi-transparent contours drawn
        """
        # Ensure mask is binary and of type uint8
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a transparent overlay
        overlay = image.copy()

        # Draw the contours on the overlay
        cv2.drawContours(overlay, contours, -1, color, thickness)

        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return result

    def draw_ellipses(self, image, mask, color=(255, 0, 0), expansion_factor=1.2, thickness=2, alpha=0.5):
        """
        Draw semi-transparent ellipses that encompass the objects identified by a binary mask on the original image.

        Args:
        image (np.array): Original image (3D numpy array for BGR images)
        mask (np.array): Binary mask (2D numpy array with values 0 or 1)
        color (tuple): BGR color tuple for the ellipses (default is green)
        expansion_factor (float): Factor to expand the ellipse size beyond the object's bounding box (default is 1.2)
        thickness (int): Thickness of the ellipse lines
        alpha (float): Transparency of the ellipses (0.0 to 1.0)

        Returns:
        np.array: Image with semi-transparent ellipses drawn
        """
        # Ensure mask is binary and of type uint8
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a transparent overlay
        overlay = image.copy()

        # Draw the ellipses on the overlay
        for contour in contours:
            # Calculate the rotated bounding box for the contour
            rect = cv2.minAreaRect(contour)
            center, axes, angle = rect

            # Calculate the expanded axes of the ellipse
            expanded_axes = (axes[0] * expansion_factor, axes[1] * expansion_factor)

            # Draw the ellipse on the overlay
            cv2.ellipse(overlay, (int(center[0]), int(center[1])), (int(expanded_axes[0] // 2), int(expanded_axes[1] // 2)), angle, 0, 360, color, thickness)

        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return result

    def draw_prompt(self, image, mask, prompt_type, color = (255, 0, 0), thickness = 2, alpha = 0.5):
        if prompt_type == "mask":
            return self.blend_image_with_colored_mask(image, mask, color, alpha)
        elif prompt_type == "bb":
            return self.draw_bounding_boxes_on_masked_regions(image, mask, color, thickness, alpha)
        elif prompt_type == "contour":
            return self.draw_contours(image, mask, color, thickness, alpha)
        elif prompt_type == "ellipse":
            return self.draw_ellipses(image=image, mask=mask, color=color, thickness=thickness, alpha=alpha)

    def get_ensamble_predictions(
        self,
        model: VipLlavaForConditionalGeneration, 
        processor: AutoProcessor, 
        support_img: np.array, 
        support_mask: np.array, 
        args
    ) -> list:
        if self.ensamble_config.is_ensamble_color_only():
            pred_class_names = []
            for color in self.ensamble_config.colors:
                prompted_image_np = self.draw_prompt(
                    support_img, 
                    support_mask, 
                    prompt_type=args.prompt_type, 
                    alpha=args.alpha_blending, 
                    color=COLORS[color], 
                    thickness=args.thickness
                )
                prompted_image = processor(text=PROMPT_TEMPLATE.format(VISUAL_PROMPTS[args.prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(model.device)
                res = model.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                pred_class_names.append(pred_class_name)

            return pred_class_names
        
        if self.ensamble_config.is_ensamble_zoom_only():
            pred_class_names = []
            for zoom_percentage in self.ensamble_config.zoom_percentages:
                prompted_image_np = self.zoom_on_masked_object(
                    self.draw_prompt(
                        support_img, 
                        support_mask, 
                        prompt_type=args.prompt_type, 
                        alpha=args.alpha_blending, 
                        color=COLORS[args.color], 
                        thickness=args.thickness
                    ), 
                    support_mask, 
                    zoom_percentage
                )
                prompted_image = processor(text=PROMPT_TEMPLATE.format(VISUAL_PROMPTS[args.prompt_type].format(args.color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(model.device)
                res = model.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                pred_class_names.append(pred_class_name)
                
            return pred_class_names
        
        if self.ensamble_config.is_ensamble_prompt_color():
            pred_class_names = []
            for prompt_type in self.ensamble_config.prompt_types:
                for color in self.ensamble_config.colors:
                    prompted_image_np = self.draw_prompt(
                        support_img, 
                        support_mask, 
                        prompt_type=prompt_type, 
                        alpha=args.alpha_blending, 
                        color=COLORS[color], 
                        thickness=args.thickness
                    )
                    prompted_image = processor(text=PROMPT_TEMPLATE.format(VISUAL_PROMPTS[prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(model.device)
                    res = model.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                    pred_class_names.append(pred_class_name)
                            
            return pred_class_names
        
        if self.ensamble_config.is_ensamble_color_zoom():
            pred_class_names = []
            for color in self.ensamble_config.colors:
                for zoom_percentage in self.ensamble_config.zoom_percentages:
                    prompted_image_np = self.zoom_on_masked_object(
                        self.draw_prompt(
                            support_img, 
                            support_mask, 
                            prompt_type=args.prompt_type, 
                            alpha=args.alpha_blending, 
                            color=COLORS[color], 
                            thickness=args.thickness
                        ), 
                        support_mask, 
                        zoom_percentage
                    )
                    prompted_image = processor(text=PROMPT_TEMPLATE.format(VISUAL_PROMPTS[args.prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(model.device)
                    res = model.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                    pred_class_names.append(pred_class_name)

            return pred_class_names
        
        if self.ensamble_config.is_ensamble_prompt_zoom():
            pred_class_names = []
            for prompt_type in self.ensamble_config.prompt_types:
                for zoom_percentage in self.ensamble_config.zoom_percentages:
                    prompted_image_np = self.zoom_on_masked_object(
                        self.draw_prompt(
                            support_img, 
                            support_mask, 
                            prompt_type=prompt_type, 
                            alpha=args.alpha_blending, 
                            color=COLORS[args.color], 
                            thickness=args.thickness
                        ), 
                        support_mask, 
                        zoom_percentage
                    )
                    prompted_image = processor(text=PROMPT_TEMPLATE.format(VISUAL_PROMPTS[prompt_type].format(args.color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(model.device)
                    res = model.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                    pred_class_names.append(pred_class_name)
                    
            return pred_class_names
        
        if self.ensamble_config.is_full_ensamble():
            pred_class_names = []
            for prompt_type in self.ensamble_config.prompt_types:
                for color in self.ensamble_config.colors:
                    for zoom_percentage in self.ensamble_config.zoom_percentages:
                        prompted_image_np = self.zoom_on_masked_object(
                            self.draw_prompt(
                                support_img, 
                                support_mask, 
                                prompt_type=prompt_type, 
                                alpha=args.alpha_blending, 
                                color=COLORS[color], 
                                thickness=args.thickness
                            ), 
                            support_mask, 
                            zoom_percentage
                        )
                        prompted_image = processor(text=PROMPT_TEMPLATE.format(VISUAL_PROMPTS[prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(model.device)
                        res = model.generate(**prompted_image, max_new_tokens=20)
                        pred_class_name = processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                        pred_class_names.append(pred_class_name)
                        
            return pred_class_names


    def fetch_semantic_info(
        self,
        support_imgs: torch.Tensor,        # (bs x ns x c x h x w)
        support_masks: torch.Tensor,       # (bs x ns x h x w)
        args,
        use_descriptions: bool = False,
        save_dir=None
    ):
        
        predicted_names = []
        for i, (s_img, s_mask) in enumerate(zip(support_imgs[0], support_masks[0])):
            support_img_numpy = np.array(transforms.ToPILImage()(s_img).convert("RGB"))
            support_mask_numpy = s_mask.numpy()
            if not self.ensamble_config.is_ensamble():
                if args.zoom_percentage > 0:
                    prompted_image_np = self.zoom_on_masked_object(
                        self.draw_prompt(
                            support_img_numpy, 
                            support_mask_numpy,
                            prompt_type=args.prompt_type,
                            alpha=args.alpha_blending,
                            color=COLORS[args.color],
                            thickness=args.thickness
                        ), 
                        support_mask_numpy, 
                        args.zoom_percentage
                    )
                else:
                    prompted_image_np = self.draw_prompt(
                        support_img_numpy, 
                        support_mask_numpy,
                        prompt_type=args.prompt_type,
                        alpha=args.alpha_blending,
                        color=COLORS[args.color],
                        thickness=args.thickness
                    )

                if save_dir is not None:
                    # SAVE PROMPTED IMAGE
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"support_prompted_{i}.png")
                    Image.fromarray(prompted_image_np).save(save_path)
                prompt = VISUAL_PROMPTS[args.prompt_type].format(args.color)
                prompt = PROMPT_TEMPLATE.format(prompt)
                prompted_image = self.vlm_processor(text=prompt, images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.vlm_model.device)
                res = self.vlm_model.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = self.vlm_processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                
                predicted_names.append(pred_class_name)
            else:
                pred_class_names = self.get_ensamble_predictions(self.vlm_model, self.vlm_processor, support_img_numpy, support_mask_numpy, args)
                counter_pred_class_names = Counter(pred_class_names)
                pred_class_name = max(counter_pred_class_names, key=counter_pred_class_names.get)
                
                predicted_names.append(pred_class_name)
        print(f"Predicted names from support images: {predicted_names}")        
        counter_predicted_names = Counter(predicted_names)
        result_name = max(counter_predicted_names, key=counter_predicted_names.get)
        
        pred_description = None
        if use_descriptions:
            if args.zoom_percentage > 0:
                prompted_image_np = self.zoom_on_masked_object(
                    self.draw_prompt(
                        support_img_numpy, 
                        support_mask_numpy,
                        prompt_type=args.prompt_type,
                        alpha=args.alpha_blending,
                        color=COLORS[args.color],
                        thickness=args.thickness
                    ), 
                    support_mask_numpy, 
                    args.zoom_percentage
                )
            else:
                prompted_image_np = self.draw_prompt(
                    support_img_numpy, 
                    support_mask_numpy,
                    prompt_type=args.prompt_type,
                    alpha=args.alpha_blending,
                    color=COLORS[args.color],
                    thickness=args.thickness
                )
            if result_name is None:
                raise ValueError("No fg_label provided")
            prompt = VISUAL_PROMPTS_DESCRIPTIONS_FROM_CLASSNAME[args.prompt_type].format(result_name, args.color)
            prompt = PROMPT_TEMPLATE.format(prompt)
            prompted_image = self.vlm_processor(text=prompt, images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.vlm_model.device)
            res = self.vlm_model.generate(**prompted_image, min_new_tokens=20, max_new_tokens=50)
            pred_description = self.vlm_processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)

            
        return result_name, pred_description

    

