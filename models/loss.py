import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from util.box_ops import box_cxcywh_to_xyxy


class ContrastiveLoss_V1_MaxQueryMatching(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, img_cond_different_txt, different_img_cond_same_txt, all_txt, logit_scale=None, return_max_ids=False):
        # img_cond_different_txt, different_img_cond_same_txt, shape = (bs, 3, n_query4m, dim)
        # all_txt, shape = (bs, 3, dim)
        logit_scale = 1 / self.temperature if logit_scale is None else logit_scale
        img_cond_different_txt, different_img_cond_same_txt, all_txt = img_cond_different_txt.half(), different_img_cond_same_txt.half(), all_txt.half()
        bsz = all_txt.shape[0]
        device = all_txt.device
        img_cond_different_txt = F.normalize(img_cond_different_txt, dim=-1)
        different_img_cond_same_txt = F.normalize(different_img_cond_same_txt, dim=-1)
        all_txt = F.normalize(all_txt, dim=-1)

        txt = all_txt[:, [0], :].unsqueeze(1) # (bs, 1, 1, dim)
        logit_t2i = different_img_cond_same_txt.matmul(txt.transpose(2, 3)).squeeze(-1) # (bs, 3, n_query4m)
        logit_t2i, t2i_max_ids = logit_t2i.max(dim=-1) # (bs, 3)
        target = torch.zeros(bsz, device=device, dtype=torch.long)
        loss_contrast_t2i = self.ce_loss(logit_t2i * logit_scale, target)
        
        all_txt = all_txt.unsqueeze(2) # (bs, 3, 1, dim)
        logit_i2t = (img_cond_different_txt * all_txt).sum(dim=-1) # (bs, 3, n_query4m)
        logit_i2t, i2t_max_ids = logit_i2t.max(dim=-1) # (bs, 3)
        loss_contrast_i2t = self.ce_loss(logit_i2t * logit_scale, target)

        if return_max_ids:
            return loss_contrast_t2i, loss_contrast_i2t, t2i_max_ids, i2t_max_ids
        return loss_contrast_t2i, loss_contrast_i2t


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, all_img, all_txt, logit_scale=None):
        logit_scale = 1 / self.temperature if logit_scale is None else logit_scale
        all_img, all_txt = all_img.half(), all_txt.half()
        bsz = all_img.shape[0]
        device = all_img.device
        all_img = F.normalize(all_img, dim=-1)
        all_txt = F.normalize(all_txt, dim=-1)

        txt = all_txt[:, [0], :]
        logit_t2i = txt.matmul(all_img.transpose(1, 2)).squeeze(1) # (bs, 3)
        target = torch.zeros(bsz, device=device, dtype=torch.long)
        loss_contrast_t2i = self.ce_loss(logit_t2i * logit_scale, target)
        
        img_pos = all_img[:, [0], :]
        logit_i2t = img_pos.matmul(all_txt.transpose(1, 2)).squeeze(1) # (bs, 3)
        loss_contrast_i2t = self.ce_loss(logit_i2t * logit_scale, target)
        return loss_contrast_t2i, loss_contrast_i2t


class MSELoss(nn.Module):
    def __init__(self, noise_scheduler, snr_gamma):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.snr_gamma = snr_gamma

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward_box_loss(self, model_pred, target_noise, timesteps, boxes):
        loss_mse = F.mse_loss(model_pred.float(), target_noise.float(), reduction="none")
        device, dtype = loss_mse.device, loss_mse.dtype
        boxes = box_cxcywh_to_xyxy(boxes) # (bs, 4)
        w_latent, h_latent = model_pred.shape[2:]
        loss_final = []
        for i, box in enumerate(boxes):
            x1, x2 = int(box[0].item() * w_latent), int(box[2].item() * w_latent)
            y1, y2 = int(box[1].item() * h_latent), int(box[3].item() * h_latent)
            if y2 > y1 and x2 > x1:
                loss_final.append(loss_mse[i, :, x1:x2, y1:y2].mean())
            else:
                loss_final.append(torch.full((1,), 0., dtype=dtype, device=device))

        loss_final = torch.stack(loss_final)
        if self.snr_gamma is None:
            loss_final = loss_final.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss_final = loss_final * mse_loss_weights
            loss_final = loss_final.mean()
        return loss_final

    def forward(self, model_pred, target_noise, timesteps):
        if self.snr_gamma is None:
            loss_mse = F.mse_loss(model_pred.float(), target_noise.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss_mse = F.mse_loss(model_pred.float(), target_noise.float(), reduction="none")
            loss_mse = loss_mse.mean(dim=list(range(1, len(loss_mse.shape)))) * mse_loss_weights
            loss_mse = loss_mse.mean()
 
        return loss_mse