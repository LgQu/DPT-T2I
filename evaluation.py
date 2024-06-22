from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import Dict, Iterable, Optional
import os
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
import json
import numpy as np
import argparse
import math
import resource

from utils import evaluate_scores, load_json
from models.postprocessors import build_postprocessors
from data.dataset.refcoco_eval import CocoEvaluator, RefExpEvaluator
from util.metrics import MetricLogger
from util.misc import targets_to
import util.dist as dist
from models.diffusion import get_noisy_latents
from models.pipeline import load_pipeline
import open_clip
from models.diffusion import get_sd_model, load_unet_lora

@torch.no_grad()
def evaluate_alignment(args, unet, loader, save_dir, device, dtype, clip=None, is_save=True, use_lora=True, 
                        num_inference_steps=20, show_tqdm=False):
    if is_save:
        save_dir = os.path.join(save_dir, f'images_generated_val_seed{args.seed}')
        os.makedirs(save_dir, exist_ok=True)
    # create pipeline
    pipeline = load_pipeline(args, unet.state_dict(), device, dtype, use_lora=use_lora)
    pipeline = pipeline.to(device)
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=True)
    # create clip
    if clip is None:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
    else:
        model, preprocess, tokenizer = clip

    # run inference
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    sims = []

    iter_loader = tqdm(loader, disable=(not show_tqdm), total=len(loader))
    for batch in iter_loader:
        ins_prompts, prompts = batch
        prompts = list(prompts)
        # generate
        pil_images = pipeline(prompts, num_inference_steps=num_inference_steps, generator=generator).images
        if is_save:
            for i_img in range(len(pil_images)):
                save_path = os.path.join(save_dir, f'{ins_prompts[i_img]:03d}' + '.jpg')
                pil_images[i_img].save(save_path)

        # cal clip score
        images = torch.stack([preprocess(img) for img in pil_images])
        texts = tokenizer(prompts)
        with torch.cuda.amp.autocast():
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(texts.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            s = (image_features * text_features).sum(dim=-1)
            sims.append(s)
    sims = torch.cat(sims)

    del pipeline
    del model, tokenizer
    torch.cuda.empty_cache()
    return sims

@torch.no_grad()
def score_batch_v1_maxQueryMatching(i, args, batch, model, noise_scheduler, encoder_hidden_states_uncond, return_unet_feat='all'):
    """
    Takes a batch of images and captions and returns a score for each image-caption pair.
    """
    diffqformer, vae, text_encoder, tokenizer = model
    imgs, texts = batch[0], batch[1]
    _, imgs_resize = imgs[0], imgs[1]
    batchsize = imgs_resize[0].shape[0]
    dtype, device = encoder_hidden_states_uncond.dtype, encoder_hidden_states_uncond.device
    scores = []
    fix_timestep = args.fix_timestep if args.fix_timestep is not None else 500
    timesteps = torch.full((batchsize,), fix_timestep, device=device)
    timesteps = timesteps.long()
    for txt_idx, text in enumerate(texts):
        noise = torch.randn((batchsize, 4, 64, 64), device=device, dtype=dtype)
        text_ids = tokenizer(list(text), max_length=tokenizer.model_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids
        encoder_hidden_states = text_encoder(text_ids.to(device))
        txt = F.normalize(encoder_hidden_states.pooler_output, dim=-1) # (bs, dim)
        txt = txt.unsqueeze(1)
        for img_idx, resized_img in enumerate(imgs_resize):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            latents = vae.encode(resized_img.to(device, dtype=dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            q_emb, _ = diffqformer(noisy_latents, timesteps, encoder_hidden_states[0], only_matching=True, return_unet_feat=return_unet_feat) # shape = (layer, bs, n_query, dim)
            img = F.normalize(q_emb[0, :, :args.num_queries_matching, :], dim=-1) # (bs, n_query_m, dim)
            s = (img * txt).sum(dim=-1) # (bs, n_query4m)
            s, _ = s.max(1)
            scores.append(s)
    scores = torch.stack(scores).permute(1, 0) if batchsize > 1 else torch.stack(scores).unsqueeze(0) # (bs, 2)
    return scores

def eval_acc(args, val_dataloader, val_dataloader2, models, noise_scheduler, encoder_hidden_states_uncond, score_batch_func):
    metrics = []
    max_more_than_onces = 0
    progress_bar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for k, batch in progress_bar_val:
        # measure time for the following line
        scores = score_batch_func(k, args, batch, models, noise_scheduler, encoder_hidden_states_uncond[[0], ...])
        acc, max_more_than_once = evaluate_scores('mscoco_val', scores, batch)
        metrics += acc
        acc = sum(metrics) / len(metrics)
        max_more_than_onces += max_more_than_once
        tqdm_logs = {'MSCOCO Val Accuracy Txt': f'{acc:.3f}', 
                    'Max more than once': f'{max_more_than_onces}'}
        progress_bar_val.set_postfix(**tqdm_logs)

    txt_acc = acc
    txt_max_more_than_onces = max_more_than_onces
    metrics = []
    max_more_than_onces = 0
    progress_bar_val2 = tqdm(enumerate(val_dataloader2), total=len(val_dataloader2))
    for k, batch in progress_bar_val2:
        # measure time for the following line
        scores = score_batch_func(k, args, batch, models, noise_scheduler, encoder_hidden_states_uncond[[0], ...])
        acc, max_more_than_once = evaluate_scores('mscoco_val', scores, batch)
        metrics += acc
        acc = sum(metrics) / len(metrics)
        max_more_than_onces += max_more_than_once
        tqdm_logs = {'MSCOCO Val Accuracy Img': f'{acc:.3f}', 
                    'Max more than once': f'{max_more_than_onces}'}
        progress_bar_val2.set_postfix(**tqdm_logs)
    img_acc = acc
    img_max_more_than_onces = max_more_than_onces
    return txt_acc, img_acc, txt_max_more_than_onces, img_max_more_than_onces

@torch.no_grad()
def eval_rec(args, models, criterion, val_tuples, weight_dict, device, dtype, return_res=False):
    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        iou_types = ["bbox"]
        evaluator_list = []
        if "refexp" in dataset_name or "refcoco" in dataset_name:
            evaluator_list.append(RefExpEvaluator(base_ds, ("bbox")))
        return evaluator_list

    def evaluate(
        models, 
        criterion: Optional[torch.nn.Module],
        contrastive_criterion: Optional[torch.nn.Module],
        qa_criterion: Optional[torch.nn.Module],
        postprocessors: Dict[str, torch.nn.Module],
        weight_dict: Dict[str, float],
        data_loader,
        evaluator_list,
        device: torch.device, 
        dtype, 
        args,
    ):  
        model, vae, noise_scheduler, tokenizer, text_encoder = models
        model.eval()
        if criterion is not None:
            criterion.eval()

        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"

        pbar = tqdm(total=len(data_loader), disable=not dist.is_main_process())
        for batch_dict in metric_logger.log_every(data_loader, int(len(data_loader) / 5), header, dist.is_main_process()):
            samples = batch_dict["samples"].to(device)
            positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
            targets = batch_dict["targets"]
            answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
            captions = [t["caption"] for t in targets]

            targets = targets_to(targets, device)

            memory_cache = None
            if args.masks:
                outputs = model(samples, captions)
            else:
                imgs, mask = samples.decompose()
                noisy_latents, timesteps, noise, _ = get_noisy_latents(args, imgs, vae, noise_scheduler, dtype, is_training=False)
                text_for_object = [t["caption"] for t in targets]
                text_for_object_ids = tokenizer(text_for_object, max_length=tokenizer.model_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids
                encoder_hidden_states = text_encoder(text_for_object_ids.to(device))[0]
                outputs = model(noisy_latents, timesteps, encoder_hidden_states, text_for_object, return_unet_feat=args.unet_feature) 

            loss_dict = {}
            if criterion is not None:
                loss_dict.update(criterion(outputs, targets, positive_map))

            if contrastive_criterion is not None:
                assert memory_cache is not None
                contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
                loss_dict["contrastive_loss"] = contrastive_loss

            if qa_criterion is not None:
                answer_losses = qa_criterion(outputs, answers)
                loss_dict.update(answer_losses)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = dist.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
            metric_logger.update(
                loss=sum(loss_dict_reduced_scaled.values()),
                **loss_dict_reduced_scaled,
                **loss_dict_reduced_unscaled,
            )

            res = None
            if not args.no_detection:
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors["bbox"](outputs, orig_target_sizes)
                if "segm" in postprocessors.keys():
                    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

                flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
                if "flickr_bbox" in postprocessors.keys():
                    image_ids = [t["original_img_id"] for t in targets]
                    sentence_ids = [t["sentence_id"] for t in targets]
                    items_per_batch_element = [t["nb_eval"] for t in targets]
                    positive_map_eval = batch_dict["positive_map_eval"].to(device)
                    flickr_results = postprocessors["flickr_bbox"](
                        outputs, orig_target_sizes, positive_map_eval, items_per_batch_element
                    )
                    assert len(flickr_results) == len(image_ids) == len(sentence_ids)
                    for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
                        flickr_res.append({"image_id": im_id, "sentence_id": sent_id, "boxes": output})

                phrasecut_res = None
                if "phrasecut" in postprocessors.keys():
                    phrasecut_res = postprocessors["phrasecut"](results)
                    assert len(targets) == len(phrasecut_res)
                    for i in range(len(targets)):
                        phrasecut_res[i]["original_id"] = targets[i]["original_id"]
                        phrasecut_res[i]["task_id"] = targets[i]["task_id"]

                res = {target["image_id"].item(): output for target, output in zip(targets, results)}

                for evaluator in evaluator_list:
                    evaluator.update(res)
            pbar.update(1)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if dist.is_main_process():
            print("Averaged stats:", metric_logger)
        for evaluator in evaluator_list:
            evaluator.synchronize_between_processes()

        refexp_res = None
        flickr_res = None
        phrasecut_res = None
        for evaluator in evaluator_list:
            if isinstance(evaluator, CocoEvaluator):
                evaluator.accumulate()
                evaluator.summarize()

            elif isinstance(evaluator, (RefExpEvaluator)):
                refexp_res = evaluator.summarize()

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        for evaluator in evaluator_list:
            if isinstance(evaluator, CocoEvaluator):
                if "bbox" in postprocessors.keys():
                    stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
                if "segm" in postprocessors.keys():
                    stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

        if refexp_res is not None:
            stats.update(refexp_res)

        if flickr_res is not None:
            stats["flickr"] = flickr_res

        if phrasecut_res is not None:
            stats["phrasecut"] = phrasecut_res

        return stats, res

    # start evaluation
    test_stats = {}
    for i, item in enumerate(val_tuples):
        evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
        item = item._replace(evaluator_list=evaluator_list)
        postprocessors = build_postprocessors(args, item.dataset_name)
        if dist.is_main_process():
            print(f"Evaluating {item.dataset_name}")
        curr_test_stats, res = evaluate(
            models=models, 
            criterion=criterion,
            contrastive_criterion=None,
            qa_criterion=None,
            postprocessors=postprocessors,
            weight_dict=weight_dict,
            data_loader=item.dataloader,
            evaluator_list=item.evaluator_list,
            device=device, 
            dtype=dtype, 
            args=args,
        )
        test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
        test_stats.update({item.dataset_name + "_" + item.split: curr_test_stats.get(item.dataset_name, None)})
    if return_res:
        return test_stats, res
    return test_stats

def load_sd_models(args):
    from my_diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    return noise_scheduler, tokenizer, text_encoder, vae, unet
