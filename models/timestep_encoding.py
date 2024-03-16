import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, GaussianFourierProjection
from diffusers.models.activations import get_activation

class TimestepEmbeddingLearned(nn.Module):
    def __init__(self, num_feats=256, time_embedding_type='positional', time_embedding_dim=None, flip_sin_to_cos=True, 
                    freq_shift=0, act_fn="silu", timestep_post_act=None, time_cond_proj_dim=None, time_embedding_act_fn=None):
        super().__init__()
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or num_feats * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or num_feats * 4
            self.time_proj = Timesteps(num_feats, flip_sin_to_cos, freq_shift)
            timestep_input_dim = num_feats
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            out_dim=num_feats, 
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

    def forward(self, timesteps, dtype=torch.float16, timestep_cond=None):
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        return emb

def build_timestep_encoding(args):
    timestep_embedding = TimestepEmbeddingLearned(num_feats=args.hidden_dim)
    return timestep_embedding

    