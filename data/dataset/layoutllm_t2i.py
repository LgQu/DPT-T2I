import torch
from torch.utils.data import Dataset

from utils import load_json

class LayoutLLM_T2I(Dataset):
    def __init__(self, json_path):
        all_prompts = []
        json_p = load_json(json_path)
        for j in json_p:
            all_prompts.append(j['caption'])
        self.all_prompts = all_prompts


    def __len__(self):
        return len(self.all_prompts)


    def __getitem__(self, index):
        return index, self.all_prompts[index].strip()
