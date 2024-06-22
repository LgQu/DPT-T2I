import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
from PIL import Image
import numpy as np
import PIL
from collections import OrderedDict
import torchvision
import os

from .transforms import diffusers_preprocess


class MSCOCODatasetHardNegtive(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, 
                hard_neg=True, neg_img=False, mixed_neg=False, tsv_path='../dataset/ITM/coco/train_hard_neg_t20_i4OpenClip_ViT-H-14.tsv', 
                latent_path=None):
        self.root_dir = root_dir
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.all_texts = self.data['title'].tolist()
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        self.mixed_neg = mixed_neg
        self.rand_neg = not self.hard_neg and not self.neg_img
        self.imgname2txtid = OrderedDict()
        for index, row in self.data.iterrows():
            if row['filepath'] not in self.imgname2txtid:
                self.imgname2txtid[row['filepath']] = [index]
            else:
                self.imgname2txtid[row['filepath']].append(index)
        self.latent_path = latent_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_file_name = row['filepath']
        img_path = f"{self.root_dir}/{img_file_name}"
        
        text = row['title']
        neg_captions =  ast.literal_eval(row['neg_caption'])
        neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]

        neg_img_paths = ast.literal_eval(row['neg_image']) # a list of row indices in self.data
        neg_paths = [f"{self.root_dir}/{i}" for i in neg_img_paths]
        
        if self.tokenizer:
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text0 = text.input_ids.squeeze(0)
            if self.mixed_neg:
                text_neg = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_neg = text_neg.input_ids.squeeze(0)
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_neg, text_rand])
            elif self.hard_neg:
                text_rand = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            elif self.rand_neg:
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            else:
                text = text0
        
        img = Image.open(img_path).convert("RGB")

        def transform(img):
            if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
            else:
                img_resize = img.resize((self.resize, self.resize))
                img_resize = diffusers_preprocess(img_resize)
            return img_resize

        imgs = [transform(img)]

        if self.latent_path is not None:
            img_latent = np.load(os.path.join(self.latent_path, img_file_name.split('.')[0] + '.npy'))
            img_latents = [img_latent]

        if self.neg_img or self.mixed_neg:
            assert not self.hard_neg
            i_neg_img = np.random.randint(0, len(neg_paths))
            img_neg_latent = np.load(os.path.join(self.latent_path, neg_img_paths[i_neg_img].split('.')[0] + '.npy'))
            img_neg = Image.open(neg_paths[i_neg_img]).convert("RGB")
            img_neg = transform(img_neg)
            imgs.append(img_neg)
            img_latents.append(img_neg_latent)

            rand_img_id = np.random.randint(0, len(self.data))
            rand_img_name = self.data.iloc[rand_img_id]['filepath']
            img_rand_latent = np.load(os.path.join(self.latent_path, rand_img_name.split('.')[0] + '.npy'))
            rand_img_path = f"{self.root_dir}/{rand_img_name}"
            img_rand = Image.open(rand_img_path).convert("RGB")
            img_rand = transform(img_rand)
            imgs.append(img_rand)
            img_latents.append(img_rand_latent)
        
        return [0, imgs], text, 0, img_latents

class ValidMSCOCODatasetHardNegtive(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=False, 
                    tsv_path='../dataset/ITM/coco/val_hard_neg_t20_i4OpenClip_ViT-H-14.tsv', neg_img=False):
        self.root_dir = root_dir
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        if not self.neg_img:
            self.hard_neg = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = f"{self.root_dir}/{img_path}"
        text = row['title']
        if self.hard_neg:
            neg_captions =  ast.literal_eval(row['neg_caption'])
            neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]
            text = [text, neg_caption]
        else:
            text = [text]

        neg_img_ids = ast.literal_eval(row['neg_image'])
        neg_paths = [f"{self.root_dir}/{i}" for i in neg_img_ids]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        return [0, imgs], text, 0
