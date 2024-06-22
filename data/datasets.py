import os.path as osp
from torchvision import datasets

from data.dataset.coco_hard_negative import MSCOCODatasetHardNegtive, ValidMSCOCODatasetHardNegtive

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch
from torch.utils.data import Dataset
from data.dataset.layoutllm_t2i import LayoutLLM_T2I


def get_dataset(dataset_name, root_dir, transform=None, resize=512, scoring_only=False, tokenizer=None, split='val', 
                max_train_samples=None, hard_neg=False, targets=None, neg_img=False, mixed_neg=False, details=False, 
                args=None, combined=False, dataset_dir='../dataset/'):
    if dataset_name == 'mscoco_hard_negative':
        root_dir = dataset_dir + 'coco/images/train2014'
        matching_dir = dataset_dir + 'ITM/coco/train_hard_neg_t20_i4OpenClip_ViT-H-14_ReC'
        if args.pretrained_model_name_or_path == 'stabilityai/stable-diffusion-2-1-base':
            latent_path = dataset_dir + 'coco/latents/train2014_SD2-1'
        elif args.pretrained_model_name_or_path == 'CompVis/stable-diffusion-v1-4':
            latent_path = dataset_dir + 'coco/latents/train2014_SD1-4'
        return MSCOCODatasetHardNegtive(root_dir, transform, resize=resize, split=split, tokenizer=tokenizer, hard_neg=hard_neg, neg_img=neg_img, 
                                        mixed_neg=mixed_neg, matching_dir=matching_dir, latent_path=latent_path)
    elif dataset_name == 'val_mscoco_hard_negative':
        root_dir = dataset_dir + 'coco/images/val2014'
        return ValidMSCOCODatasetHardNegtive(root_dir, transform, resize=resize, split='val', tokenizer=tokenizer, neg_img=neg_img, hard_neg=hard_neg)
    elif dataset_name in ['refcoco', 'refcoco+', 'refcocog', 'refall']:
        root_dir = dataset_dir + 'coco/images'
        ann_path = dataset_dir + 'ReC/mdetr/OpenSource'
        ann_caption_file = dataset_dir + 'coco/annotations/captions_train2014.json'
        if combined:
            from .dataset.refcoco import build_dataset_combined as build_dataset
            extra_args = {'matching_dir': f'{dataset_dir}ITM/coco/{split}_hard_neg_t20_i4OpenClip_ViT-H-14'}
        else:
            from .dataset.refcoco import build_dataset
            extra_args = {}

        if split == 'train':
            if args.pretrained_model_name_or_path == 'stabilityai/stable-diffusion-2-1-base':
                extra_args['latent_path'] = dataset_dir + 'coco/latents/train2014_SD2-1'
            elif args.pretrained_model_name_or_path == 'CompVis/stable-diffusion-v1-4':
                extra_args['latent_path'] = dataset_dir + 'coco/latents/train2014_SD1-4'
            else:
                raise ValueError(f'Unknown pre-trained model: {args.pretrained_model_name_or_path}')
            dataset = build_dataset(args, 'train', root_dir, dataset_name, ann_path, ann_caption_file, 
                                    test=False, test_type='', masks=False, **extra_args)
        elif split == 'val':
            dataset = build_dataset(args, 'val', root_dir, dataset_name, ann_path, ann_caption_file, 
                                    test=True, test_type='val', masks=False, **extra_args)
        elif split in ['testA', 'testB', 'test']:
            dataset = build_dataset(args, 'test', root_dir, dataset_name, ann_path, ann_caption_file, 
                                    test=True, test_type=split, masks=False, **extra_args)
        else:
            raise ValueError(f'Unknown split {split}')
        return dataset
    elif dataset_name == 'LayoutLLM-T2I':
        if split in ['val', 'text']:
            return LayoutLLM_T2I(args.prompts_path_val)
        else:
            raise ValueError(f'Unknown split {split}')
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
