from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast
import collections
import numpy as np

from .transformer import build_transformer
from .position_encoding import build_position_encoding
from .timestep_encoding import build_timestep_encoding
from utils import NestedTensor

def build_model(args, backbone):
    transformer = build_transformer(args)
    position_embedding = build_position_encoding(args)
    timestep_embedding = build_timestep_encoding(args)

    if args.pretrained_model_name_or_path == 'stabilityai/stable-diffusion-2-1-base':
        clip_dim = 1024
    elif args.pretrained_model_name_or_path == 'CompVis/stable-diffusion-v1-4':
        clip_dim = 768
    else:
        raise ValueError(f'Unknown pre-trained model: {args.pretrained_model_name_or_path}')

    model = DPT(
        backbone,
        position_embedding, 
        timestep_embedding, 
        transformer,
        num_classes=255,
        num_queries_matching=args.num_queries_matching,
        num_queries_rec=args.num_queries_rec, 
        aux_loss=False,
        contrastive_align_loss=args.contrastive_align_loss, 
        contrastive_hdim=args.contrastive_loss_hdim, 
        text_encoder_type=args.text_encoder_type, 
        freeze_text_encoder=args.freeze_text_encoder, 
        have_matching_head=True, 
        unet_feature=args.unet_feature, 
        clip_dim=clip_dim
    )
    return model

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DPTmatching(nn.Module):
    def __init__(self, backbone, position_embedding, timestep_embedding, transformer, num_classes, num_queries, 
                    aux_loss=False, have_matching_head=False, clip_dim = 1024, unet_feature='mid'):
        nn.Module.__init__(self)
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.unet_feature = unet_feature
        self.feature_idx_map = {'bottom0': 0, 'bottom1': 1, 'bottom2': 2, 'bottom3': 3, 
                                    'mid': 4, 'up0': 5, 'up1': 6, 'up2': 7, 'up3': 8}
        self.backbone_num_channels_map = collections.OrderedDict({'bottom0': 320, 'bottom1': 640, 'bottom2': 1280, 'bottom3': 1280, 
                                    'mid': 1280, 'up0': 1280, 'up1': 1280, 'up2': 640, 'up3': 320, 
                                    'bottom2_bottom3_mid': 1280 * 3, 'bottom1_up0': 1280 + 640, 
                                    'bottom0_up1': 1280 + 320})
        self.num_features_fusion = 9 
        self.down_sampling = nn.Identity()

        backbone_num_channels = self.backbone_num_channels_map.get(unet_feature, None)
        if 'all' in unet_feature and backbone_num_channels is None: # e.g., all-1280-8
            backbone_num_channels = int(unet_feature.split('-')[1])
            channels_fus = list(self.backbone_num_channels_map.values())[:self.num_features_fusion]
            self.fusion_proj = nn.ModuleList([nn.Conv2d(c, backbone_num_channels, kernel_size=1)
                                                for c in channels_fus])

        self.input_proj = nn.Conv2d(backbone_num_channels, hidden_dim, kernel_size=1)
        self.unet = backbone
        self.aux_loss = aux_loss
        self.position_embedding = position_embedding
        self.timestep_embedding = timestep_embedding
        if have_matching_head:
            self.map_to_clip = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, clip_dim))
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def get_return_feat(self, unet_feat):
        if '_' in unet_feat:
            fs = unet_feat.split('_')
            ins_map = [self.feature_idx_map[f] for f in fs]
            return fs[np.argmax(ins_map)]
        elif 'all' in unet_feat and '-' in unet_feat:
            return 'all'
        return unet_feat

    def forward_enc_dec(self, noisy_latents, timesteps, encoder_hidden_states, return_unet_feat='all', detach_unet_feat=None, 
                        added_cond_kwargs=None):
        noise_pred, bottom_mid_up_feature_maps = self.unet(noisy_latents, timesteps, encoder_hidden_states, 
                                                            return_unet_feat=self.get_return_feat(return_unet_feat), 
                                                            detach_unet_feat=detach_unet_feat, 
                                                            added_cond_kwargs=added_cond_kwargs)
        if '_' in self.unet_feature:
            fs = self.unet_feature.split('_')
            feature_maps = [bottom_mid_up_feature_maps[self.feature_idx_map[f]] for f in fs]
            feature_maps = torch.cat(feature_maps, dim=1)
        elif 'all' in self.unet_feature:
            assert len(bottom_mid_up_feature_maps) == self.num_features_fusion
            size_fus = int(self.unet_feature.split('-')[2])
            feature_maps = 0
            for i, f in enumerate(bottom_mid_up_feature_maps):
                f = F.interpolate(f, size=(size_fus, size_fus))
                feature_maps = feature_maps + self.fusion_proj[i](f)
        else:
            feature_maps = bottom_mid_up_feature_maps[self.feature_idx_map[self.unet_feature]]

        # downsampling
        feature_maps = self.down_sampling(feature_maps)

        b, c, h, w = feature_maps.shape
        device = feature_maps.device
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        pos = self.position_embedding(NestedTensor(feature_maps, mask))
        time_emb = self.timestep_embedding(timesteps, dtype=feature_maps.dtype)
        emb = pos + time_emb[:, :, None, None]
        src = feature_maps
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, emb)[0]
        return hs, noise_pred.sample
    
    def state_dict(self, ):
        new_state = collections.OrderedDict()
        old_state = super().state_dict()
        named_parameters = dict(self.named_parameters())
        for k, v in old_state.items():
            if k in named_parameters and named_parameters[k].requires_grad: 
                new_state[k] = v
        return new_state
    
    def load_state_dict_qformer(self, local_state_dict, device='cpu'):
        local_state_dict = torch.load(local_state_dict, map_location=device)
        for name, child in self.named_children():
            # print(name)
            if name not in ['unet', 'text_encoder']:
                child_state_dict = {k.lstrip(name)[1:]: v for k, v in local_state_dict.items() if k.startswith(name)}
                child.load_state_dict(child_state_dict)
    
    def load_unet_lora(self, local_state_dict):
        local_state_dict = torch.load(local_state_dict)
        lora_state_dict = collections.OrderedDict()
        for k, v in local_state_dict.items():
            name_sub = k.split('.')
            if name_sub[0] == 'unet' and (name_sub[8] == 'processor' or name_sub[7] == 'processor'):
                lora_state_dict[k] = v
        self.unet.load_attn_procs(lora_state_dict)

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DPT(DPTmatching):
    def __init__(self, backbone, position_embedding, timestep_embedding, transformer, num_queries_matching, 
                    num_queries_rec, contrastive_align_loss=True, num_classes=255, contrastive_hdim=64, 
                    text_encoder_type='roberta-base', aux_loss=False, freeze_text_encoder=True, 
                    have_matching_head=True, unet_feature='mid', clip_dim=1024):
        num_queries = num_queries_matching + num_queries_rec
        super(DPT, self).__init__(backbone, position_embedding, timestep_embedding, transformer, num_classes, num_queries, aux_loss, 
                                        have_matching_head=have_matching_head, unet_feature=unet_feature, clip_dim=clip_dim)
        hidden_dim = transformer.d_model
        self.num_queries_matching = num_queries_matching
        self.num_queries_rec = num_queries_rec
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.contrastive_align_loss = contrastive_align_loss
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(self.text_encoder.config.hidden_size, contrastive_hdim)
 
    def _forward_matching(self, query_out, output_dict=False):
        q_emb = self.map_to_clip(query_out)
        if output_dict:
            return {'q_emb': q_emb, 'logit_scale': self.logit_scale.exp()}
        return q_emb, self.logit_scale.exp()

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, text=None, only_matching=False, return_unet_feat='all', 
                return_dict=False, detach_unet_feat=None, added_cond_kwargs=None):
        device = noisy_latents.device
        query_out, noise_pred = self.forward_enc_dec(noisy_latents, timesteps, encoder_hidden_states, return_unet_feat=return_unet_feat, 
                                                        detach_unet_feat=detach_unet_feat, added_cond_kwargs=added_cond_kwargs) 
        query_out_matching = query_out[:, :, :self.num_queries_matching, :]
        # global matching
        q_emb, logit_scale = self._forward_matching(query_out_matching)
        out = {"noise_pred": noise_pred, "q_emb_matching": q_emb, "logit_scale": logit_scale}

        if only_matching:
            if return_dict:
                return out
            else:
                return q_emb, logit_scale

        # local grounding
        query_out_rec = query_out[:, :, -self.num_queries_rec:, :]
        outputs_class = self.class_embed(query_out_rec)
        outputs_coord = self.bbox_embed(query_out_rec).sigmoid()
        
        out.update(
            {
                "query_out_rec": query_out_rec[-1], 
                "pred_logits": outputs_class[-1], # -1 means the last layer
                "pred_boxes": outputs_coord[-1],
            }
        )

        proj_queries, proj_tokens = None, None
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_image(query_out_rec), p=2, dim=-1)
            tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_memory = encoded_text.last_hidden_state
            proj_tokens = F.normalize(self.contrastive_align_projection_text(text_memory), p=2, dim=-1)
            out.update(
                {   "tokenized": tokenized, 
                    "proj_queries": proj_queries[-1],
                    "proj_tokens": proj_tokens, 
                    "caption": text
                }
            )
        return out