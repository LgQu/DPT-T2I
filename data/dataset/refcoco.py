from pathlib import Path
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from transformers import RobertaTokenizerFast
from functools import partial
import sys
import os
sys.path.append('./')
from pycocotools.coco import COCO
import numpy as np
from collections import OrderedDict
import pandas as pd
import ast
from PIL import Image
from transformers import CLIPTokenizer

import util
import data.dataset.transforms as T
from .transforms import diffusers_preprocess
from utils import load_json

def get_coco_api_from_dataset(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if isinstance(dataset, (torchvision.datasets.CocoDetection)):
        return dataset.coco
    else:
        raise ValueError(f'Unknown dataset type: {type(dataset)}')


def make_coco_transforms(image_set, cautious):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    if image_set == "train":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, max_size, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=max_size),
                normalize,
            ]
        )
    raise ValueError(f"unknown {image_set}")


def make_diff_rec_transforms(new_size):
    return T.Compose([T.Resize(new_size), T.DiffNormalize()])


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_tokens=False, tokenizer=None):
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None
        caption_whole_image = target["caption_whole_image"] if "caption_whole_image" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        area = torch.tensor([obj["area"] for obj in anno])
        
        # guard against no boxes via resizing
        boxes[:, 2:] += boxes[:, :2] # (x, y, w, h) --> (x1, y1, x2, y2)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        tokens_positive = [] if self.return_tokens else None
        if self.return_tokens and anno and "tokens" in anno[0]:
            tokens_positive = [obj["tokens"] for obj in anno]
        elif self.return_tokens and anno and "tokens_positive" in anno[0]:
            tokens_positive = [obj["tokens_positive"] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if caption_whole_image is not None:
            target["caption_whole_image"] = caption_whole_image
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if tokens_positive is not None:
            target["tokens_positive"] = []

            for i, k in enumerate(keep):
                if k:
                    target["tokens_positive"].append(tokens_positive[i])

        if isfinal is not None:
            target["isfinal"] = isfinal

        # for conversion to coco api
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens and self.tokenizer is not None:
            assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt")
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
        return image, target

class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, ann_caption_file, return_masks, return_tokens, tokenizer, transforms, is_train=False):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)
        self.is_train = is_train
        self.coco_cap = COCO(ann_caption_file)


    def __getitem__(self, idx):
        img, target = super(ModulatedDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        image_id_ori = self.coco.imgs[image_id]['original_id']
        assert self.coco.imgs[image_id]['file_name'] == self.coco_cap.imgs[image_id_ori]['file_name']
        captions_whole_img = [ann["caption"] for ann in self.coco_cap.loadAnns(self.coco_cap.getAnnIds(image_id_ori))]
        caption_whole_img_selected = captions_whole_img[np.random.randint(0, len(captions_whole_img))]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        target = {"image_id": image_id, "annotations": target, "caption": caption, 'caption_whole_image': caption_whole_img_selected}
        img, target = self.prepare(img, target)
        img, target = self._transforms(img, target)

        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]

        if "tokens_positive_eval" in coco_img and not self.is_train:
            tokenized = self.prepare.tokenizer(caption, return_tensors="pt")
            target["positive_map_eval"] = create_positive_map(tokenized, coco_img["tokens_positive_eval"])
            target["nb_eval"] = len(target["positive_map_eval"])

        return img, target

class MatchingReCCombined(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, ann_caption_file, return_masks, return_tokens, tokenizer, transforms, 
                    clip_tokenizer, latent_path=None, is_train=False, matching_dir='../dataset/ITM/coco/train_hard_neg_t20_i4OpenClip_ViT-H-14'):
        super(MatchingReCCombined, self).__init__(img_folder, ann_file)
        self.img_folder = img_folder
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)
        self.clip_tokenizer = clip_tokenizer
        self.is_train = is_train
        assert is_train, 'Other split settings are not implemented!'
        self.latent_path = latent_path
        if is_train: 
            self.matching_dir = matching_dir
            self.imgname2txtid = load_json(os.path.join(self.matching_dir, 'imgname2txtid.json'))
            all_txt_ids = []
            for k, v in self.imgname2txtid.items():
                all_txt_ids.extend(v)
            self.num_txt = max(all_txt_ids)
        else:
            self.coco_cap = COCO(ann_caption_file)

    def __getitem__(self, idx):
        img, target = super(MatchingReCCombined, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_file_name = self.coco.imgs[image_id]['file_name']
        negatives = None
        if self.is_train:
            # -- Global Matching --
            pos_txt_ids = self.imgname2txtid[img_file_name]
            pos_txt_id_selected = pos_txt_ids[np.random.randint(0, len(pos_txt_ids))]
            row = load_json(os.path.join(self.matching_dir, f'{pos_txt_id_selected:05d}.json'))
            caption_whole_img_selected = row['title']
            # neg_caption
            neg_captions =  ast.literal_eval(row['neg_caption'])
            neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]
            clip_max_length = self.clip_tokenizer.model_max_length
            text0 = self.clip_tokenizer(caption_whole_img_selected, max_length=clip_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids.squeeze(0)
            text_neg = self.clip_tokenizer(neg_caption, max_length=clip_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids.squeeze(0)
            text_rand = load_json(os.path.join(self.matching_dir, f'{np.random.randint(0, self.num_txt):05d}.json'))['title']
            text_rand = self.clip_tokenizer(text_rand, max_length=clip_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids.squeeze(0)
            # neg_image
            neg_img_paths = ast.literal_eval(row['neg_image']) # a list of row indices in self.data
            neg_paths = [f"{self.img_folder}/{i}" for i in neg_img_paths]
            i_neg_img = np.random.randint(0, len(neg_paths))
            img_neg = Image.open(neg_paths[i_neg_img]).convert("RGB")
            img_neg, _ = self._transforms(img_neg, None)
            rand_img_id = self.ids[np.random.randint(0, len(self.ids))]
            rand_img_name = self.coco.imgs[rand_img_id]['file_name']
            rand_img_path = f"{self.img_folder}/{rand_img_name}"
            img_rand = Image.open(rand_img_path).convert("RGB")
            img_rand, _ = self._transforms(img_rand, None)
            negatives = {'text_neg': text_neg, 'text_rand': text_rand, 'img_neg': img_neg, 'img_rand': img_rand}
            if self.latent_path is not None:
                img_neg_latent = np.load(os.path.join(self.latent_path, neg_img_paths[i_neg_img].split('.')[0] + '.npy'))
                negatives['img_neg_latent'] = torch.tensor(img_neg_latent)
                img_rand_latent = np.load(os.path.join(self.latent_path, rand_img_name.split('.')[0] + '.npy'))
                negatives['img_rand_latent'] = torch.tensor(img_rand_latent)
        else:
            image_id_ori = self.coco.imgs[image_id]['original_id']
            assert self.coco.imgs[image_id]['file_name'] == self.coco_cap.imgs[image_id_ori]['file_name']
            captions_whole_img = [ann["caption"] for ann in self.coco_cap.loadAnns(self.coco_cap.getAnnIds(image_id_ori))]
            caption_whole_img_selected = captions_whole_img[np.random.randint(0, len(captions_whole_img))]

        # -- ReC --
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        target = {"image_id": image_id, "annotations": target, "caption": caption, 'caption_whole_image': caption_whole_img_selected}
        img, target = self.prepare(img, target)
        img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]
        if "tokens_positive_eval" in coco_img and not self.is_train:
            tokenized = self.prepare.tokenizer(caption, return_tensors="pt")
            target["positive_map_eval"] = create_positive_map(tokenized, coco_img["tokens_positive_eval"])
            target["nb_eval"] = len(target["positive_map_eval"])
        if self.latent_path is not None:
            img_latent = np.load(os.path.join(self.latent_path, img_file_name.split('.')[0] + '.npy'))
            target["img_latent"] = torch.tensor(img_latent)

        return img, target, negatives


class RefExpDetection(ModulatedDetection):
    pass


def build_dataset(args, image_set, coco_path, refexp_dataset_name, refexp_ann_path, ann_caption_file, test, test_type, masks=False):
    img_dir = Path(coco_path) / "train2014"

    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        if test:
            test_set = test_type
            ann_file = Path(refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        else:
            ann_file = Path(refexp_ann_path) / f"finetune_{refexp_dataset_name}_{image_set}.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = RefExpDetection(
        img_dir,
        ann_file,
        ann_caption_file, 
        return_masks=masks,
        return_tokens=True,
        tokenizer=tokenizer,
        transforms=make_diff_rec_transforms((args.resolution, args.resolution)), # make_coco_transforms(image_set, cautious=True),
    )
    return dataset


def build_dataset_combined(args, image_set, coco_path, refexp_dataset_name, refexp_ann_path, ann_caption_file, test, test_type, 
                            masks=False, tsv_path='../dataset/ITM/coco/train_hard_neg_t20_i4OpenClip_ViT-H-14.tsv', latent_path=None):
    img_dir = Path(coco_path) / "train2014"

    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog", "refall"]:
        if test:
            test_set = test_type
            ann_file = Path(refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        else:
            ann_file = Path(refexp_ann_path) / f"finetune_{refexp_dataset_name}_{image_set}.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    dataset = MatchingReCCombined(
        img_dir,
        ann_file,
        ann_caption_file, 
        return_masks=masks,
        return_tokens=True,
        tokenizer=tokenizer,
        clip_tokenizer=clip_tokenizer, 
        transforms=make_diff_rec_transforms((args.resolution, args.resolution)), # make_coco_transforms(image_set, cautious=True),
        tsv_path=tsv_path, 
        is_train = not test, 
        latent_path = latent_path
    )
    return dataset
