
import torch
from pytorch_lightning import seed_everything
import psutil
import json
import os
import zipfile
from typing import NamedTuple, List, Callable
import numpy as np
from typing import Optional, List
from torch import Tensor

RETRIEVAL_TASKS = ['imagecode', 'imagecode_video', 'flickr30k', 'imagenet', 'clevr', 'svo_verb', 'svo_subj', 
                    'svo_obj', 'pets', 'flickr30k_text', 'vg_relation', 'vg_attribution', 'coco_order', 
                    'flickr30k_order', 'mscoco_val']


def report_inconsistent_ckpt(accelerator, model, ckpt_dir, logger):
    if accelerator.is_main_process:
        raw_model_ckpt = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location='cpu')
        model_unwrap = accelerator.unwrap_model(model)
        new_model_ckpt = model_unwrap.state_dict()
        info = {}
        for k, v in new_model_ckpt.items():
            if k not in raw_model_ckpt:
                info[k] = 'new'
            elif k in raw_model_ckpt and v.shape != raw_model_ckpt[k].shape:
                info[k] = f'{raw_model_ckpt[k].shape} --> {v.shape}'
        for k, v in raw_model_ckpt.items():
            if k not in new_model_ckpt:
                info[k] = 'missing'
        if len(info) > 0:
            logger.info(f'Inconsistent ckpt info: {info}')

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

class Box(NamedTuple):
    x: int
    y: int
    w: int = 0
    h: int = 0

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return Box(self.x + self.w // 2, self.y + self.h // 2)

    def corners(self):
        yield Box(self.x, self.y)
        yield Box(self.x + self.w, self.y)
        yield Box(self.x + self.w, self.y + self.h)
        yield Box(self.x, self.y + self.h)

    @property
    def area(self):
        return self.w * self.h

    def intersect(self, other: "Box") -> "Box":
        x1 = max(self.x, other.x)
        x2 = max(x1, min(self.x+self.w, other.x+other.w))
        y1 = max(self.y, other.y)
        y2 = max(y1, min(self.y+self.h, other.y+other.h))
        return Box(x=x1, y=y1, w=x2-x1, h=y2-y1)

    def min_bounding(self, other: "Box") -> "Box":
        corners = list(self.corners())
        corners.extend(other.corners())
        min_x = min_y = float("inf")
        max_x = max_y = -float("inf")

        for item in corners:
            min_x = min(min_x, item.x)
            min_y = min(min_y, item.y)
            max_x = max(max_x, item.x)
            max_y = max(max_y, item.y)

        return Box(min_x, min_y, max_x - min_x, max_y - min_y)

    def expand(self, growth: float = .1) -> "Box":
        factor = 1 + growth
        w = factor * self.w
        h = factor * self.h
        return Box(min_x - (w - self.w) / 2, min_y - (h - self.h) / 2, w, h)

def iou(box1, box2):
    x1 = max(box1.x, box2.x)
    x2 = max(x1, min(box1.x+box1.w, box2.x+box2.w))
    y1 = max(box1.y, box2.y)
    y2 = max(y1, min(box1.y+box1.h, box2.y+box2.h))
    intersection = Box(x=x1, y=y1, w=x2-x1, h=y2-y1)
    intersection_area = intersection.area
    union_area = box1.area+box2.area-intersection_area
    return intersection_area / union_area

def evaluate_retrieval(task, scores, img_idx):
    # print(scores, img_idx)
    if type(scores) != list:
        img_idx = img_idx.cpu().numpy()
        scores = scores.cpu().numpy()
    scores = np.stack(scores, axis=0)
    retrieval_accuracy = []
    max_more_than_once = 0
    # print(scores.shape)
    # print(img_idx.shape)
    for i in range(scores.shape[0]):
        number_of_argmax_appearances = np.sum(scores[i] == np.max(scores[i]))
        if number_of_argmax_appearances > 1:
            max_more_than_once += 1
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)
    # R5 calculation too
    if task in ['flickr30k', 'imagecode', 'imagenet', 'flickr30k_text']:
        r5 = []
        for i in range(scores.shape[0]):
            if img_idx[i] in np.argsort(scores[i])[-5:]:
                r5.append(1)
            else:
                r5.append(0)
        return retrieval_accuracy, r5, max_more_than_once
    else:
        return retrieval_accuracy, max_more_than_once

def evaluate_gender_bias(m_attr_scores, f_attr_scores, class_ids):
    entity = class_ids[0].split('_')[-1] # either clothes, drinks, or bags
    male_filter = np.array(class_ids)==f'male_{entity}' # indices of scores of male images
    female_filter = np.array(class_ids)==f'female_{entity}' # indices of scores of female images
    m_attr_scores = m_attr_scores.cpu().numpy() # all the images scored with the male attr word
    f_attr_scores = f_attr_scores.cpu().numpy() # all the images scores w female attr word
    
    m_imgs_m_attr = m_attr_scores[male_filter]
    m_imgs_f_attr = f_attr_scores[male_filter]
    f_imgs_m_attr = m_attr_scores[female_filter]
    f_imgs_f_attr = f_attr_scores[female_filter]
    
    phi_male = m_imgs_m_attr - m_imgs_f_attr #phi(m,w_m,w_f) = sigma(m,w_m)-sigma(m,w_f)
    phi_female = f_imgs_m_attr - f_imgs_f_attr #phi(f,w_m,w_f) = sigma(f,w_m)-sigma(f,w_f)
    
    return {f'male_{entity}':phi_male,f'female_{entity}':phi_female}

def evaluate_winoground(scores):
    text_score, img_score, group_score = [], [], []
    for score_ in scores:
        c0_i0, c0_i1, c1_i0, c1_i1 = score_
        text_score_ = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
        img_score_ = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
        group_score_ = 1 if text_score_ and img_score_ else 0 
        text_score.append(text_score_)
        img_score.append(img_score_)
        group_score.append(group_score_)
    return text_score, img_score, group_score

def evaluate_bias(good_scores, bad_scores, img_idx):
    img_idx = img_idx.cpu().numpy()
    good_scores = good_scores.cpu().numpy()
    bad_scores = bad_scores.cpu().numpy()
    phis = {}
    for i in range(len(good_scores)): # rows of tensor are images, columns are the words
        # p val test just needs the phi(w,A,B) which i have!  just code it elionrrr
        class_idx = int(img_idx[i]) # get class, should be an integer {0,1,...,7}
        good_score = good_scores[i].mean() # mean_{a\in A} sigma(x,a)
        bad_score = bad_scores[i].mean() # mean_{b\in B} sigma(x,b)
        phi = good_score-bad_score # phi(w,A,B) = mean_{a\in A} sigma(x,a) - mean_{b\in B} sigma(x,b)
        if class_idx in phis:
            phis[class_idx].append(phi)
        else:
            phis[class_idx] = [phi]
    return phis#, raw_scores

def evaluate_scores(task, scores, batch):
    if task == 'winoground':
        score = evaluate_winoground(scores)
    elif task == 'mmbias':
        # so we have a bunch of scores, which is a tensor Size([batchsize,len(texts)])
        # example for 4 texts and batchsize 2
        # scores = tensor([[ 0.0555,  0.0121,  0.0113,mmOKxRfPbYjE -0.0000],
        #         [ 0.0398, -0.0133, -0.0340, -0.0391]], device='cuda:7')
        text_len = floor(len(batch[1])/2) # number of good / bad texts
        good_scores = scores[:, :text_len]  # extract the first len(good_texts) cols for pleasant_texts
        bad_scores = scores[:, text_len:]   # extract the remaining cols for unpleasant_texts
        assert len(good_scores) == len(bad_scores)
        img_idx = batch[-1] # tensor of class_ids
        return evaluate_bias(good_scores, bad_scores, img_idx) # dictionary of lists of phis
    elif task == 'genderbias':
        # input is list of scores (tensors whatever), ill use batchsize 6 so its just one text and one fe/male_entity
        # evaluate_gender_bias should return a just the phi for the one class
        male_attr_scores = scores[:,0]
        female_attr_scores = scores[:,-1]
        class_ids = batch[-1]
        return evaluate_gender_bias(male_attr_scores, female_attr_scores, class_ids) 
    elif task in RETRIEVAL_TASKS:
        img_idx = batch[-1]
        score = evaluate_retrieval(task, scores, img_idx)
    else:
        raise NotImplementedError
    return score