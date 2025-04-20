<p align="center">
  <h2 align="center" style="margin-top: -30px;">MLLMs Know Where to Look: <br>Training-free Perception of Small Visual Details with Multimodal LLMs</h2>
</p>

<div style="font-family: charter;" align="center">
    <a href="https://saccharomycetes.github.io/" target="_blank">Jiarui Zhang</a>,
    <a href="https://mahyarkoy.github.io/" target="_blank">Mahyar Khayatkhoei</a>,
    <a href="https://www.prateekchhikara.com/" target="_blank">Prateek Chhikara</a>,
    <a href="https://www.ilievski.info/" target="_blank">Filip Ilievski</a>
</div>

<br>

<p align="center">
  <img src="images/method_case.png" alt="Method Overview" width="600">
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2502.17422-b31b1b.svg)](https://arxiv.org/abs/2502.17422)
[![OpenReview](https://img.shields.io/badge/OpenReview-DgaY5mDdmT-blue.svg)](https://openreview.net/forum?id=DgaY5mDdmT)
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-green.svg)](https://iclr.cc/Conferences/2025)
[![Dataset](https://img.shields.io/badge/Dataset-orange.svg)](https://huggingface.co/datasets/jrzhang/TextVQA_GT_bbox)

## 📰 News 
- [2025-04-11] We have released the [code](qwen2_5_methods.py) for running our method on Qwen-2.5-VL, and the TextVQA groundtruth bounding box dataset for the analysis in our paper (find below).
- [2025-03-24] Qwen-2.5-VL [implementation](qwen2_5_implementation.ipynb) is added, thanks to [@zenjieli](https://github.com/zenjieli).
- [2025-01-26] We release our code.
- [2025-01-21] Our paper is accepted by ICLR 2025!

## 📦 To be released
- New results on Qwen-2.5-VL
- And more, please stay tuned!

## 📋 Overview

This repository contains the official implementation of our ICLR 2025 paper "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs". Our method enables multimodal large language models (MLLMs) to better perceive small visual details without any additional training. This repository provides the detailed implementation of applying our methods on multiple MLLMs and benchmark datasets. 

## 🔥 Highlights

- 🔍 We find that MLLMs often know where to look, even if their answers are wrong.
- 📸 We propose a training-free method to significantly enhance MLLMs' visual perception on small visual details.
- 💪 Our method is flexible with different visual inputs formats, including high-resolution images (see below), multiple images, and video (to be explored in the future).

### Running Qwen-2.5-VL

```bash
bash run_all.sh textvqa qwen2_5 rel_att
```

You can adjust the attention layers [here](qwen2_5_methods.py#L9), and the model resolution [here](run.py#L211).

### TextVQA ground truth bounding boxes

Through [link](https://huggingface.co/datasets/jrzhang/TextVQA_GT_bbox), or directly download with:

```python
from datasets import load_dataset

ds = load_dataset("jrzhang/TextVQA_GT_bbox")['train']
```


## 🛠️ Installation

### Setup Environment
```bash
# Create and activate conda environment
conda create -n mllms_know python=3.10
conda activate mllms_know

# Install dependencies
pip install -r requirements.txt

# Install modified transformers library
cd transformers
pip install -e .
cd ..
```

## 🚀 Quick Start

We provide a [quick start notebook](quick_start.ipynb) that demonstrates how to:
- Load and process images
- Apply our methods to enhance visual perception
- Visualize attention maps

## 📊 Benchmark Evaluation

### Dataset Preparation
1. Download the benchmark datasets and corresponding images to your local directory
2. Update the paths in `info.py` with your local directory paths

Example (textvqa)

Dataset preparation:
```bash
mkdir -p data/textvqa/images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P data/textvqa/images
unzip data/textvqa/images/train_val_images.zip -d data/textvqa/images
rm data/textvqa/images/train_val_images.zip
mv data/textvqa/images/train_images/* data/textvqa/images
rm -r data/textvqa/images/train_images
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P data/textvqa
```

Dataset processing (to a unified format):
```python
import json

with open('data/textvqa/TextVQA_0.5.1_val.json') as f:
    datas = json.load(f)

new_datas = []
for data_id, data in enumerate(datas['data']):
    data_id = str(data_id).zfill(10)
    question = data['question']
    labels = data['answers']
    image_path = f"{data['image_id']}.jpg"
    new_data = {
        'id': data_id,
        'question': question,
        'labels': labels,
        'image_path': image_path
    }
    new_datas.append(new_data)

with open('data/textvqa/data.json', 'w') as f:
    json.dump(new_datas, f, indent=4)
```


### Running Evaluations
To run our method on benchmark datasets, use the provided script:

```bash
# Format: bash run_all.sh [dataset] [model] [method]
bash run_all.sh textvqa llava rel_att
```

Get the model's performance:
```bash
python get_score.py --data_dir ./data/results --save_path ./
```

### Datasets Links
- [TextVQA](https://textvqa.org/dataset/)
- [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=downloads) (Task 1)
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- [AOKVQA](https://github.com/allenai/aokvqa?tab=readme-ov-file#downloading-the-dataset)
- [POPE](https://huggingface.co/datasets/lmms-lab/POPE)
- [VSTAR](https://huggingface.co/datasets/craigwu/vstar_bench)
- [VQAv2](https://visualqa.org/download.html)

### Models
- LLaVA-1.5 (`llava`)
- InstructBLIP (`blip`)

For implementation details, see `llava_methods.py` and `blip_methods.py`. Please feel free to explore other MLLMs!

## 📝 Method Details

Our approach leverages inherent attention mechanisms and gradients in MLLMs to identify regions of interest without additional training. The key methods include:

1. **Relative Attention-based Visual Cropping**: Computes relative attention \(A_{rel}(x,q)\) for each image-question pair and selects a target layer from TextVQA validation data to guide visual cropping.

2. **Gradient-Weighted Attention-based Visual Cropping**: Uses gradient information to refine attention maps, normalizing answer-to-token and token-to-image attention without requiring a second forward pass.

3. **Input Gradient-based Visual Cropping**: Directly computes the gradient of the model’s decision w.r.t. the input image. To mitigate noise in uniform regions, it applies Gaussian high-pass filtering, median filtering, and thresholding before spatial aggregation.


**Bounding Box Selection for Visual Cropping.**  
We use a sliding window approach to extract bounding boxes from the importance map. Windows of different sizes, scaled by factors in $\{1, 1.2, \dots, 2\}$, slide over the image with a stride of 1. The position maximizing the sum of importance values is selected, and the window with the largest deviation from its neighbors is chosen. The cropped region is then resized and fed into the MLLM.

**High-Resolution Visual Cropping.**  
For high-resolution images ($>1K$), we first split them into smaller non-overlapping blocks ($<1024\times1024$), compute importance maps for each block, and merge them. The same bounding box selection is then applied to the merged importance map.


For implementation details, see `llava_methods.py` and `blip_methods.py` and `utils.py`.

## 📊 Results

Our method significantly improves MLLMs' performance on tasks requiring perception of small visual details, such as text recognition in images, fine-grained object recognition, and spatial reasoning. Please refer to the [paper](https://arxiv.org/abs/2502.17422) for more details and run the demo notebook for better understanding!

## 📚 Citation

If you find our paper and code useful for your research and applications, please cite using this BibTeX:
```bibtex 
@inproceedings{
  zhang2025mllms,
  title={{MLLM}s Know Where to Look: Training-free Perception of Small Visual Details with Multimodal {LLM}s},
  author={Jiarui Zhang and Mahyar Khayatkhoei and Prateek Chhikara and Filip Ilievski},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2502.17422}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
