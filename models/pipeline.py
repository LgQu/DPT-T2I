from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch
import numpy as np
import wandb

def load_pipeline(args, unet_state_dict, device, dtype, use_lora=True):
    unet_new = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # freeze parameters of models to save more memory
    unet_new.requires_grad_(False)
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet_new.to(device, dtype=dtype)

    if use_lora:
        lora_attn_procs = {}
        for name in unet_new.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet_new.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet_new.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet_new.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet_new.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, 
                                                        rank=args.lora_rank)

        unet_new.set_attn_processor(lora_attn_procs)
        unet_new.load_state_dict(unet_state_dict)

    # unregister_unet_output(unet)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet_new,
        revision=args.revision,
        torch_dtype=dtype,
        safety_checker=None, 
        requires_safety_checker=False
    )


    pipeline.enable_xformers_memory_efficient_attention()
    return pipeline

def generate(args, accelerator, unet, epoch, dtype, use_lora=True):
    # create pipeline
    pipeline = load_pipeline(args, unet.state_dict(), accelerator.device, dtype, use_lora=use_lora)
    pipeline = pipeline.to(accelerator.device)
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    images = []
    all_prompts = []
    for i in range(len(args.validation_prompts)):
        prompts_duplicated = [args.validation_prompts[i]] * args.num_validation_images
        all_prompts.extend(prompts_duplicated)
        # with torch.autocast("cuda"):
        images_for_each_prompt = pipeline(prompts_duplicated, num_inference_steps=20, generator=generator).images
        images.extend(images_for_each_prompt)
    
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {all_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
    del pipeline
    torch.cuda.empty_cache()