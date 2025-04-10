import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *

# currently select 22 but feel free to try other layers
ATT_LAYER = 22

def rel_attention_qwen2_5(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)

    outputs = model(**inputs, output_attentions=True)
    general_outputs = model(**general_inputs, output_attentions=True)

    att = outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    general_att = general_outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    att_map = att / general_att

    att_map = att_map.reshape(att_shape)

    return att_map