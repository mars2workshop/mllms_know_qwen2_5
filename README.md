# mllms_know_qwen2_5
This project is based on mllms_know and uses qwen2.5 for inference experiments. It is part of the ICCV2025 MARS2 workshop extended experiments.

## Install

Please follow the environment setup instructions from [mllms_know](https://github.com/saccharomycetes/mllms_know) 

**Do not** perform the transformers installation step.. Instead, use the following command to install the required version of transformers:

```
pip install transformers==4.51.0
```

## inference
### Single GPU Run
If you have only one GPU, simply run `running.py`.

### Multi-GPU Run
If you want to run the experiment on multiple GPUs, run `myrun.sh`.

`myrun.sh` also applies to the single GPU scenario, and we strongly recommend using this approach in both cases.


### Custom Parameters
#### Single GPU

you can change parameters in `running.py`

#### Multi-GPU
For multi-GPU setup, modify the following parameters in `myrun.sh`:

**gpus**: Specify the GPU IDs, for example: (0,1,2,3,4,5,6,7).

Modify the following parameters in `running_sh.py`:

**model_path**: Path to the model weights (if you want to download from HuggingFace directly, modify this path).

**question_path**: Path to the VQA-SA-question file, for example: `model_path ="path/to/VQA-SA-question.json".`

**image_dict**: Path to the VQA-SA-images directory, for example: `question_path ="path/to/VQA-SA-images/".`

**max_pixels**: The desired image resolution, which must be a multiple of 28x28.

**out_path**: The path for output results, for example:`out_path = f"path/to/result_chunk_{chunk_id}.json"`

### Output
It will output a number of json files corresponding to the number of GPUs, all in the required standard data format. Please merge them manually.

### Note
Please **DO NOT** use flash-attn!
