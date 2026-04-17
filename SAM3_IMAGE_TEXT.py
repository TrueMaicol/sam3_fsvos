import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3_IMAGE_PREDICTOR():
    def __init__(self, checkpoint=None):
        if checkpoint is None:
            raise Exception('checkpoint is required')
        self.checkpoint = checkpoint

        self.model = build_sam3_image_model(checkpoint_path=self.checkpoint)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)

        # Enable bfloat16 autocast for inference, exactly as done in the official SAM3 notebooks.
        self._autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self._autocast_ctx.__enter__()

    def extract_image_from_state(self, inference_state=None):
        if inference_state is None:
            raise Exception('Inference State is None')
        
        print(f"Found {len(inference_state['scores'])} objects")
        
        # Debug: Check both masks and masks_logits
        # print(f"Shape of inference_state['masks']: {inference_state['masks'].shape}")      
        
        # Merge all object masks into a single mask
        merged_mask = np.any(np.array(inference_state['masks'].cpu()), axis=0)
        # print(f"Final merged mask has shape {merged_mask.shape}")

        return merged_mask.squeeze(0)
            

    def prompt_text(self, image=None, text_prompt=None):
        if image is None:
            raise Exception('Image is None')
        elif text_prompt is None: 
            raise Exception('Text Prompt is None')
        
        inference_state=self.processor.set_image(image)
        
        inference_state = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        # print(f"Inference state keys: {inference_state.keys()}")
        return self.extract_image_from_state(inference_state)

    def prompt_text_with_box(self, image=None, text_prompt=None, box=None):
        if image is None:
            raise Exception('Image is None')
        elif text_prompt is None: 
            raise Exception('Text Prompt is None')
        elif box is None:
            raise Exception("Box Prompt is None")

        # add the box and text prompt inside the inference state and then manually trigger the forward image
        inference_state = self.processor.set_image(image)
        # 2. Prepare Text Concept (Only runs the Language encoder)
        text_outputs = self.processor.model.backbone.forward_text(
            [text_prompt], device=self.processor.device
        )
        inference_state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in inference_state:
            # Initialize a dummy geometric state if it doesn't exist
            inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

        # Convert box to [batch, seq, 4] and labels to [batch, seq]
        boxes = torch.tensor(box, device=self.processor.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([True], device=self.processor.device, dtype=torch.bool).view(1, 1)
        inference_state["geometric_prompt"].append_boxes(boxes, labels)
        
        # This triggers the Transformer Decoder with both Text + Box conditioning
        inference_state = self.processor._forward_grounding(inference_state)
        
        return self.extract_image_from_state(inference_state)
        
    def prompt_text_with_points(self, image=None, text_prompt=None, points=None):
        if image is None:
            raise Exception('Image is None')
        elif text_prompt is None: 
            raise Exception('Text Prompt is None')
        elif points is None:
            raise Exception("Points Prompt is None")

        # add the points and text prompt inside the inference state and then manually trigger the forward image
        inference_state = self.processor.set_image(image)
        # 2. Prepare Text Concept (Only runs the Language encoder)
        text_outputs = self.processor.model.backbone.forward_text(
            [text_prompt], device=self.processor.device
        )
        inference_state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in inference_state:
            # Initialize a dummy geometric state if it doesn't exist
            inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

        # Convert points to [seq, batch, 2] and labels to [seq, batch] (batch-second format)
        # points arrives as (N, 1, 2) from main.py's convert_points_format
        points = torch.tensor(points, device=self.processor.device, dtype=torch.float32)
        num_points = points.shape[0]
        labels = torch.ones((num_points, 1), device=self.processor.device, dtype=torch.bool)
        inference_state["geometric_prompt"].append_points(points, labels)
        
        # This triggers the Transformer Decoder with both Text + Points conditioning
        inference_state = self.processor._forward_grounding(inference_state)
        
        return self.extract_image_from_state(inference_state)

    def prompt_point(self, image=None, points=None, point_labels=None):
        if image is None:
            raise Exception('Image is None')
        elif points is None:
            raise Exception("Point Prompt is None")
        elif point_labels is None:
            raise Exception("Point Labels is None")

        # add the point and text prompt inside the inference state and then manually trigger the forward image
        inference_state = self.processor.set_image(image)

        if "language_features" not in inference_state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.processor.model.backbone.forward_text(
                ["visual"], device=self.processor.device
            )
            inference_state["backbone_out"].update(dummy_text_outputs)
        
        if "geometric_prompt" not in inference_state:
            # Initialize a dummy geometric state if it doesn't exist
            inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

        # Convert points to [seq, batch, 2] and labels to [seq, batch] (batch-second format)
        # points arrives as (N, 1, 2) from main.py's convert_points_format
        points = torch.tensor(points, device=self.processor.device, dtype=torch.float32)
        num_points = points.shape[0]
        labels = torch.tensor(point_labels, device=self.processor.device, dtype=torch.bool)
        inference_state["geometric_prompt"].append_points(points, labels)
        
        # This triggers the Transformer Decoder with both Text + Points conditioning
        inference_state = self.processor._forward_grounding(inference_state)
        
        return self.extract_image_from_state(inference_state)
        


