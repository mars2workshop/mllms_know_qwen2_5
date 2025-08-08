import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import argparse

def main(args):
    """
    total_chunks: =gpu_nums
    chunk_id:
    """
    device = 'cuda'
    total_chunks = args.total_chunks
    chunk_id = args.chunk_id

    model_path = "/home/zhoujiakai/tmp/load"
    question_path = "/home/zhoujiakai/tmp/VQA-SA-question copy.json"
    image_dict = "/home/zhoujiakai/tmp/VQA-SA-images/"
    max_pixels = 256 * 28 * 28
    out_path = f"/home/zhoujiakai/tmp/result_chunk_{chunk_id}.json"
  
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path,max_pixels=max_pixels)

    from run import vicrop_qa
    import json

    with open(question_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    output_data = []
    model_name = 'qwen2_5'
    method_name = 'rel_att'

    avg_chunk_size = len(input_data) // total_chunks
    remainder = len(input_data) % total_chunks

    chunks = []
    start_idx = 0

    start_idx = chunk_id * avg_chunk_size + min(chunk_id, remainder)
    end_idx = start_idx + avg_chunk_size + (1 if chunk_id < remainder else 0)
    chunk = input_data[start_idx:end_idx]

    for idx,item in enumerate(chunk):
        print(f"Processed {idx} items so far...")
        image_path_ori = item["image_path"]
        image_path = image_path_ori.replace("images\\", image_dict)
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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    args = parser.parse_args()
    main(args)

