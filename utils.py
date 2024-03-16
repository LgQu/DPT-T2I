
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