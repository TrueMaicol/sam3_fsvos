import sys
sys.path.append("/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src")
import torch
from SAM3_IMAGE_TEXT import SAM3_IMAGE_PREDICTOR
sam3 = SAM3_IMAGE_PREDICTOR(checkpoint="/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/sam3.1_checkpoint/sam3.1_multiplex.pt")
image = torch.zeros((3, 518, 518), dtype=torch.float32)
state = sam3.processor.set_image(image)
text_outputs = sam3.processor.model.backbone.forward_text(["cat"], device=sam3.processor.device)
state["backbone_out"].update(text_outputs)
state["geometric_prompt"] = sam3.processor.model._get_dummy_prompt()
prompt, prompt_mask, backbone_out = sam3.processor.model._encode_prompt(    backbone_out=state["backbone_out"],    find_input=sam3.processor.find_stage,    geometric_prompt=state["geometric_prompt"],    encode_text=True,    skip_coords=False)
backbone_out, encoder_out, _ = sam3.processor.model._run_encoder(    backbone_out, sam3.processor.find_stage, prompt, prompt_mask)
print("Encoder hidden states shape:", encoder_out["encoder_hidden_states"].shape)