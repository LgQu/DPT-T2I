# DPT-T2I
<h3><a href="">Discriminative Probing and Tuning for Text-to-Image Generation</a></h3>

[Leigang Qu](https://leigang-qu.github.io/), [Wenjie Wang](https://wenjiewwj.github.io/), [Yongqi Li](https://liyongqi67.github.io/), [Hanwang Zhang](https://personal.ntu.edu.sg/hanwangzhang/), [Liqiang Nie](https://liqiangnie.github.io/), and [Tat-Seng Chua](https://www.chuatatseng.com/)

<a href='https://dpt-t2i.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2403.04321'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This repository contains code and links to  the DPT model for text-to-image (T2I) generation to improve text-image alignment. We show the potential of discriminative abilities of pre-trained T2I models and significant gains on text-image alignment after discriminative tuning based on image-text matching (ITM) and referring expression comprehension (REC) tasks.  



## Introduction

Schematic illustration of the proposed **discriminative probing and tuning (DPT)** framework. We first extract semantic representations from the frozen SD and then propose a discriminative adapter to conduct discriminative probing to investigate the global matching and local grounding abilities of SD. Afterward, we perform parameter-efficient discriminative tuning by introducing LoRA parameters. During inference, we present the self-correction mechanism to guide the denoising-based text-to-image generation.

![](assets/framework.png)



## Release

- [x] Release the training code. 
- [x] Release the inference code and [checkpoint (v2.1)](https://huggingface.co/leigangqu/DPT-T2I/resolve/main/pytorch_model.bin?download=true) for text-to-image synthesis. 
- [x] Release the paper of DPT on [arXiv](https://arxiv.org/pdf/2403.04321.pdf). 



## Installation

The requirements file has the dependencies that are needed by DPT-T2I. The following is the instruction how to install dependencies. 

First, clone the repository locally: 

```bash
git clone https://github.com/LgQu/DPT-T2I.git
```

Make a new conda env and activate it:

```bash
conda create -n dpt_t2i python=3.8
conda activate dpt_t2i
```

Install the the packages in the requirements.txt:

```bash
pip install -r requirements.txt
```



## Text-to-Image Synthesis

First, download the [checkpoint (v2.1)](https://huggingface.co/leigangqu/DPT-T2I/resolve/main/pytorch_model.bin?download=true) for LoRA based on [Stable Diffusion (v2.1)](https://huggingface.co/stabilityai/stable-diffusion-2-1), and then put it in the directory `./ckpt/dpt-v2.1`. 

Run `txt2img.py`: 

```python
python txt2img.py --gpuid 0 --prompt "a painting of a virus monster playing guitar"
```

The generated images can be seen in the `./outputs` directory. 



## Training

In **stage 1**, we freeze the UNet of Stable Diffusion and train the discriminative adapter: 

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port=255487 train_stage1.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --unet_feature='up1' \
    --dataloader_num_workers=4 \
    --center_crop --random_flip \
    --lr_scheduler="constant"  \
    --checkpointing_steps=500 \
    --train_batch_size=48 \
    --val_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=60000 \
    --learning_rate=1e-4  \
    --run_name='dpt_stage1' \
    --report_to=wandb
```

where  `MODEL_NAME` can be `"stabilityai/stable-diffusion-2-1-base"` or `"CompVis/stable-diffusion-v1-4"`. 



In **stage 2**, we train the whole model including the UNet of Stable Difffusion (with LoRA) and the discriminative adapter:

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port=25548 train_stage2.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --unet_feature='up1' \
    --lora_rank=4 \
    --mse_loss_coef=1 \
    --dataloader_num_workers=8 \
    --center_crop --random_flip \
    --lr_scheduler="cosine"  \
    --checkpointing_steps=200 \
    --train_batch_size=8 \
    --val_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --max_train_steps=5000 \
    --learning_rate=1e-4 \
    --qformer_ckpt $CKPT_STAGE1 \
    --run_name='dpt_stage2' \
    --report_to=wandb
```

where `$CKPT_STAGE1` denotes the directory of the checkpoint obtained in stage 1. 



## Citation

If you find our work useful in your research, please consider citing DPT:

```tex
@inproceedings{qu2024discriminative,
  title={Discriminative Probing and Tuning for Text-to-Image Generation},
  author={Qu, Leigang and Wang, Wenjie and Li, Yongqi and Zhang, Hanwang and Nie, Liqiang and Chua, Tat-Seng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7434--7444},
  year={2024}
}
```

## Acknowledgement

We thank the authors of [DETR](https://github.com/facebookresearch/detr), [MDETR](https://github.com/ashkamath/mdetr), and [DiffusionITM](https://github.com/McGill-NLP/diffusion-itm), for making their code available. 
