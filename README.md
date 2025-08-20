# Intruduction
This repository provides the implementation of the method proposed in the ICLR 2025 paper 'MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs' on the Qwen2.5-VL model, along with the testing of our VQA-SA dataset."

---

## Install

Please follow the environment setup instructions from ["mllms_know"](https://github.com/saccharomycetes/mllms_know) 

**Do not** perform the transformers installation step. Instead, use the following command to install the required version of transformers:

```
pip install transformers==4.51.0
```

---

## How to Run Inference
### Prepare Input JSON
The file should be a list of entries in this format:`VQA-SA-question.json`
```
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?"
  },
  ...
]
```
### Run Script
Please set the following configuration options, then run `myrun.sh`.

`myrun.sh`:

**gpus**: Specify the GPU IDs, for example: (0 1 2 3 4 5 6 7).

`running_sh.py`:

**model_path**: Path to the model weights (if you want to download from HuggingFace directly, modify this path).

**question_path**: Path to the VQA-SA-question file, for example: `model_path ="path/to/VQA-SA-question.json".`

**image_dict**: Path to the VQA-SA-images directory, for example: `question_path ="path/to/VQA-SA-images/".`

**max_pixels**: The desired image resolution, which must be a multiple of 28x28.

**out_path**: The path for output results, for example:`out_path = f"path/to/result_chunk_{chunk_id}.json"`

### Output Format
The result will be saved as JSON files containing th answer for each input. The number of JSON files will be the same as the number of GPUs used. Please merge them.
```
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?",
    "result": "..."
  },
  ...
]

---

### Note
Please **DO NOT** use flash-attn!

---

## Citation
If you find this code and our data useful for your research and applications, please cite using this BibTeX:
```
@inproceedings{
  zhang2025mllms,
  title={{MLLM}s Know Where to Look: Training-free Perception of Small Visual Details with Multimodal {LLM}s},
  author={Jiarui Zhang and Mahyar Khayatkhoei and Prateek Chhikara and Filip Ilievski},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2502.17422}
}
```
```
@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}
```
```
@article{yao2025lens,
title={LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models},
author={Yao, Ruilin and Zhang, Bo and Huang, Jirui and Long, Xinwei and Zhang, Yifang and Zou, Tianyu and Wu, Yufei and Su, Shichao and Xu, Yifan and Zeng, Wenxi and others},
journal={arXiv preprint arXiv:2505.15616},
year={2025}
}
```

---

## Acknowledge
This codebase is partially based on ["mllms_know"](https://github.com/saccharomycetes/mllms_know) 

