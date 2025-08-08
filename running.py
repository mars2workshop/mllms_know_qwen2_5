import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

device = 'cuda'
model_path = "/home/zhoujiakai/tmp/load"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(device)
#processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left', use_fast=True)
max_pixels = 256 * 28 * 28
processor = AutoProcessor.from_pretrained(model_path,max_pixels=max_pixels)

from run import vicrop_qa
import json

with open("/home/zhoujiakai/tmp/VQA-SA-question copy.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

output_data = []
model_name = 'qwen2_5'
method_name = 'rel_att'

for idx,item in enumerate(input_data):
    print(f"Processed {idx} items so far...")
    image_path_ori = item["image_path"]
    image_path = image_path_ori.replace("images\\", "/home/zhoujiakai/tmp/VQA-SA-images/")
    question = item["question"]
    short_question = question

    A,result,B,C = vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question)
    #print(result)

    output_item = {
            "image_path": image_path_ori,
            "question": question,
            "result": result
        }
    output_data.append(output_item)

with open("/home/zhoujiakai/tmp/result.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)