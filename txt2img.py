import os
from typing import Union
from tqdm import tqdm
import random
import argparse
import collections

from pytorch_lightning import seed_everything
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler

from models.model import build_model_combined
from models.unet import register_unet_output, unregister_unet_output


def load_unet_lora(unet, local_state_dict, device='cpu'):
    print(f'Load lora ckpt from {local_state_dict}')
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


@torch.no_grad()
def init_prompt(prompt, pipeline, return_pooler=False):
    uncond_input = pipeline.tokenizer(
        [""] * len(prompt), padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))
    text_input = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))
    context = torch.cat([uncond_embeddings[0], text_embeddings[0]])
    if return_pooler:
        return context, torch.cat([uncond_embeddings[1], text_embeddings[1]])
    return context

def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = timestep, min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999)
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def generate_with_correction(pipeline, model, ddim_scheduler, steps, prompts, guidance_scale=7.5, 
                                correction_factor=0.1, args=None):
    ddim_scheduler.set_timesteps(args.num_timesteps)
    bs = len(prompts)
    device = pipeline.unet.device   
    context, context_pooled = init_prompt(prompts, pipeline, return_pooler=True)
    latent = torch.randn((bs, 4, 64, 64), device=device, dtype=context.dtype)
    all_latent = [latent]
    for i in tqdm(range(steps), leave=False):
        t = ddim_scheduler.timesteps[i]
        x_in = torch.cat([latent] * 2)
        t_in = torch.stack([t] * bs * 2).to(device)
        
        with torch.set_grad_enabled(True) and torch.cuda.amp.autocast():
            register_unet_output(model.unet)
            txt = F.normalize(context_pooled[bs:], dim=-1) 
            txt = txt.unsqueeze(1)
            x_in.requires_grad_(True)
            q_emb, _ = model(x_in[bs:], t_in[bs:], context[bs:], only_matching=True) 
            img = F.normalize(q_emb[0, :, :args.num_queries_matching, :], dim=-1) 
            s = (img * txt).sum(dim=-1) 
            s, _ = s.max(1) 
            grad = torch.autograd.grad(s.sum(), x_in)[0]

        with torch.no_grad():
            x_in = x_in.detach() + correction_factor * grad
            unregister_unet_output(model.unet)
            noise_uncond, noise = pipeline.unet(x_in, t_in, encoder_hidden_states=context)["sample"].chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise - noise_uncond)
            latent = next_step(noise_pred, t, latent, ddim_scheduler)
            all_latent.append(latent)

    with torch.no_grad():
        # decode
        latent = all_latent[-2]
        latent = 1 / 0.18215 * latent
        image = pipeline.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a evaluation script.")
    parser.add_argument("-g", "--gpu", type=str, default='0')
    parser.add_argument("--prompt", type=str, default="a painting of a virus monster playing guitar", 
        help="the prompt to render")
    parser.add_argument("--n_samples", type=int, default=3, 
        help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--seed", type=int, default=42, 
        help="random seed")
    parser.add_argument("--dtype", type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2-1-base', 
        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--unet_feature", type=str, default='up1', 
        help='unet feature extracted from the probing layer')
    parser.add_argument('--correction_factor', default=0.5, type=float,
        help="guidance factor (eta) in self-correction")
    parser.add_argument('--num_timesteps', default=50, type=int)
    parser.add_argument("--outdir", type=str, nargs="?", default="./outputs", 
        help="dir to write results to")
    parser.add_argument("--revision", type=str, default=None, required=False, 
        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--resolution", type=int, default=512)
    # model
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=1, type=int,
        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries_matching', default=10, type=int)
    parser.add_argument('--num_queries_rec', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--fix_timestep', default=None, type=int, 
        help="use one timestep to train the model")
    parser.add_argument("--no_contrastive_align_loss", dest="contrastive_align_loss", action="store_false",
        help="Whether to add contrastive alignment loss")
    parser.add_argument("--contrastive_loss_hdim", type=int, default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss")
    parser.add_argument('--text_encoder_type', type=str, default='roberta-base')
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false", 
        help="Whether to freeze the weights of the text encoder")
    parser.add_argument("--masks", action="store_true") # for segmentation
    parser.add_argument("--no_detection", action="store_true", 
        help="Whether to train the detector")
    args = parser.parse_args()


    model_ckpt = "./ckpt/dpt-v2.1/pytorch_model.bin" 
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.seed)
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    # -------------------------------- load model --------------------------------
    unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    load_unet_lora(unet, model_ckpt, device)
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        revision=args.revision,
        torch_dtype=dtype,
        safety_checker=None, 
        requires_safety_checker=False
    )
    if args.correction_factor > 0:
        model = build_model_combined(args, unet, is_inference=True)      
        model.to(device, dtype=dtype)
        model.load_state_dict_qformer(model_ckpt, device=device)

    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.unet = unet
    pipe = pipe.to(device)
    ddim_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    prompt_batch = [args.prompt] * args.n_samples

    os.makedirs(args.outdir, exist_ok=True)
    base_count = len(os.listdir(args.outdir))

    if args.correction_factor > 0:
        pil_images = generate_with_correction(pipe, model, ddim_scheduler, args.num_timesteps, prompt_batch, 
                                                correction_factor=args.correction_factor, args=args)
    else:
        pil_images = pipe(prompt_batch, num_inference_steps=args.num_timesteps).images

    for i_img in range(len(pil_images)):
        pil_images[i_img].save(os.path.join(args.outdir, f"{base_count:05}.png"))
        base_count += 1
