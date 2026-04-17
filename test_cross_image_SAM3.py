from SAM3_IMAGE_TEXT import SAM3_IMAGE_PREDICTOR
from PIL import Image
import numpy as np
import torch 

def save_image(image, path):
    """Save a numpy array or PIL image to the specified path."""
    if torch.is_tensor(image):
        image = image.numpy()
    if isinstance(image, np.ndarray):
        # Handle boolean arrays (masks)
        if image.dtype == bool:
            image = (image * 255).astype(np.uint8)
        # Handle float32 images from transform
        elif image.dtype == np.float32:
            # Only multiply by 255 if values are in 0-1 range
            # Masks from ADE20K/PASCAL have values 0-255, don't multiply again
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image)
    image.save(path)


if __name__ == "__main__":
    sam3_image_predictor = SAM3_IMAGE_PREDICTOR(checkpoint="/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src/checkpoints/sam3.pt")
    
    # 5-Shot Support images and masks
    support_images_paths = [
        "/leonardo_work/IscrC_MARSv2/datasets/cross_image_test/reference.png",
        # Add more paths for 5-shot
    ]
    support_masks_paths = [
        "/leonardo_work/IscrC_MARSv2/datasets/cross_image_test/reference_mask.png",
        # Add more paths for 5-shot
    ]
    target_image_path = "/leonardo_work/IscrC_MARSv2/datasets/cross_image_test/target.png"
    text_prompt = "an object" # Replace with your class name, e.g., "a black dog"
    
    all_visual_tokens = []
    all_visual_masks = []
    
    # 1. Process Support Images (Purely Visual)
    for i, (img_path, mask_path) in enumerate(zip(support_images_paths, support_masks_paths)):
        print(f"\nProcessing Support Image {i+1}: {img_path}")
        img = Image.open(img_path)
        mask = Image.open(mask_path).resize((sam3_image_predictor.processor.resolution, sam3_image_predictor.processor.resolution), Image.NEAREST)
        mask_arr = np.array(mask) > 0
        
        y_coords, x_coords = np.where(mask_arr)
        if len(x_coords) == 0:
            print(f"Warning: Mask for {img_path} is empty. Skipping.")
            continue
            
        # Sample 20 points
        indices = np.random.choice(len(x_coords), min(20, len(x_coords)), replace=False)
        pts = np.stack([x_coords[indices] / mask_arr.shape[1], y_coords[indices] / mask_arr.shape[0]], axis=1)
        
        # Set image and encode visual-only tokens
        state = sam3_image_predictor.processor.set_image(img)
        state = sam3_image_predictor.processor.add_point_prompts(pts, [True]*len(pts), state)
        state = sam3_image_predictor.processor._encode_current_prompts(state, encode_text=False)
        
        # save the prompt tokens (prompt_mask is just a mask to indicate which tokens are valid)
        all_visual_tokens.append(state["prompt"])
        all_visual_masks.append(state["prompt_mask"])

    if not all_visual_tokens:
        raise ValueError("No valid support shots found.")

    # Aggregate visual knowledge
    aggregated_visual_prompt = torch.cat(all_visual_tokens, dim=0)
    aggregated_visual_mask = torch.cat(all_visual_masks, dim=1)
    
    # 2. Predict Target Image with Text Fusion
    print(f"\nPredicting Target Image with Text Fusion: '{text_prompt}'")
    target_image = Image.open(target_image_path)
    state_target = sam3_image_predictor.processor.set_image(target_image)
    
    # Generate real text tokens from target backbone
    text_outputs = sam3_image_predictor.processor.model.backbone.forward_text(
        [text_prompt], device=sam3_image_predictor.processor.device
    )
    # text_outputs['language_features'] is usually [L, 1, C]
    # text_outputs['language_mask'] is usually [1, L]
    txt_tokens = text_outputs["language_features"][:, [0]] # select first text prompt
    txt_mask = text_outputs["language_mask"][[0]]
    
    # 3. Combine Text + Multi-Shot Visuals
    final_prompt = torch.cat([txt_tokens, aggregated_visual_prompt], dim=0)
    final_mask = torch.cat([txt_mask, aggregated_visual_mask], dim=1)
    
    # update the state_target with the prompt_tokens from the support images and the text tokens
    state_target["prompt"] = final_prompt
    state_target["prompt_mask"] = final_mask
    state_target["backbone_out"].update(text_outputs)

    # run the inference using the updated state
    state_target_inference = sam3_image_predictor.processor._forward_with_encoded_prompt(state_target)
    print(f"Detected {len(state_target_inference.get('masks', []))} objects in target image.")
    save_image(sam3_image_predictor.extract_image_from_state(state_target_inference), "/leonardo_work/IscrC_MARSv2/datasets/cross_image_test/target_prediction.png")
