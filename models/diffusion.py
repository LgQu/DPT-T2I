import torch
import time
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, \
    EulerDiscreteScheduler, DDIMScheduler, StableDiffusionAttendAndExcitePipeline
from transformers import CLIPTextModel, CLIPTokenizer
import collections

def load_unet_lora(unet, local_state_dict, device='cpu'):
    local_state_dict = torch.load(local_state_dict, map_location=device)
    try:
        unet.load_attn_procs(local_state_dict)
    except:
        lora_state_dict = collections.OrderedDict()
        for k, v in local_state_dict.items():
            name_sub = k.split('.')
            if name_sub[0] == 'unet' and (name_sub[8] == 'processor' or name_sub[7] == 'processor'):
                lora_state_dict[k] = v
        unet.load_attn_procs(lora_state_dict)

def get_sd_model(args):
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError
    device = args.device

    if args.version in ['1-4', '1-1', '1-2']:
        model_id = f"CompVis/stable-diffusion-v{args.version}"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        if args.other_pipe is not None:
            if args.other_pipe == 'attend_and_excite':
                pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, 
                                                        safety_checker=None, requires_safety_checker=False)
            else:
                raise ValueError(f'Unimplemented pipi: {args.other_pipe}')
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, 
                                                        safety_checker=None, requires_safety_checker=False)
        

        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        if model_id != args.model_id:
            load_unet_lora(pipe.unet, args.model_id, device)
        unet = pipe.unet
        
    elif args.version == '2-1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        if args.other_pipe is not None:
            if args.other_pipe == 'attend_and_excite':
                pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)
            else:
                raise ValueError(f'Unimplemented pipi: {args.other_pipe}')
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)

        pipe.enable_xformers_memory_efficient_attention()
        
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        if model_id != args.model_id:
            load_unet_lora(pipe.unet, args.model_id, device)
                        
            unet = pipe.unet
        else:
            unet = pipe.unet
        pipe.vae = vae
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer

    else:
        raise NotImplementedError

    return vae, tokenizer, text_encoder, unet, scheduler, pipe


def get_scheduler_config(args):
    assert args.version in {'1-1', '1-2', '1-4', '2-1'}
    if args.version == '1-4':
        # https://github.com/huggingface/diffusers/issues/960
        config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
    elif args.version == '2-1':
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,  # todo
            "trained_betas": None
        }
    elif args.version in ['1-1', '1-2']:
        config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None
            }
    else:
        raise NotImplementedError

    return config

def get_noisy_latents(args, imgs, vae, noise_scheduler, dtype, noise_and_t=None, is_training=True, latents=None):
    if latents is None:
        latents = vae.encode(imgs.to(dtype=dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    if noise_and_t is not None:
        noise, timesteps = noise_and_t
    else:
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        if args.fix_timestep is None and is_training:
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        else:
            fix_timestep = args.fix_timestep if args.fix_timestep is not None else int(noise_scheduler.config.num_train_timesteps / 2)
            timesteps = torch.full((bsz,), fix_timestep, device=latents.device)
        timesteps = timesteps.long()
    
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    return noisy_latents, timesteps, noise, target