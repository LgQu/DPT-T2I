import argparse
import logging
import math
import os
import random
from pathlib import Path
from functools import partial

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from datetime import datetime
import resource  
from collections import namedtuple
from accelerate import DistributedDataParallelKwargs

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from data.datasets import get_dataset
from utils import report_inconsistent_ckpt
from models.unet import register_unet_output, unregister_unet_output
from models.model import build_model_combined
from models.loss import ContrastiveLoss
from models.diffusion import get_noisy_latents
from evaluation import eval_acc, eval_rec
from models.pipeline import load_pipeline, generate
import util
from data.dataset.refcoco import get_coco_api_from_dataset

from models.loss import ContrastiveLoss_V1_MaxQueryMatching as ContrastiveLossClass
from evaluation import score_batch_v1_maxQueryMatching as score_batch_func


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# it is a solution to solve the problem "RuntimeError: received 0 items of ancdata", refer to https://github.com/pytorch/pytorch/issues/973#issuecomment-345088750
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE) # get the number of file descriptors the current process can open
print('resource.RLIMIT_NOFILE : {}'.format(resource.getrlimit(resource.RLIMIT_NOFILE)))
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
print('reset resource.RLIMIT_NOFILE to: {}'.format(resource.getrlimit(resource.RLIMIT_NOFILE)))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='refall',
        required=True, 
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--dataset_script",
        type=str,
        default=None,
        help=(
            "Dataset script path to custom one dataset"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=['An astronaut is riding a horse.'],
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1337, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--noise_offset", 
        type=float, 
        default=0, 
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default='', 
        help="Run name"
    )    
    # DiffusionITM
    parser.add_argument(
        '--neg_prob', 
        type=float, 
        default=1.0, 
        help='The probability of sampling a negative image.'
    )
    parser.add_argument(
        '--img_root', 
        type=str, 
        default='../dataset/coco/images'
    )
    parser.add_argument(
        '--hard_neg', 
        action='store_true'
    )
    parser.add_argument(
        '--relativistic', 
        action='store_true'
    )
    parser.add_argument(
        '--unhinged', 
        action='store_true'
    )
    parser.add_argument(
        '--neg_img', 
        action='store_true'
    )
    parser.add_argument(
        '--mixed_neg', 
        action='store_true'
    )
    # DiffQformer
    # - Backbone
    parser.add_argument(
        '--position_embedding', 
        default='sine', 
        type=str, 
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )
    # - Transformer
    parser.add_argument(
        '--enc_layers', 
        default=1, 
        type=int, 
        help="Number of encoding layers in the transformer"
    )
    parser.add_argument(
        '--dec_layers', 
        default=1, 
        type=int, 
        help="Number of decoding layers in the transformer"
    )
    parser.add_argument(
        '--dim_feedforward', 
        default=2048, 
        type=int, 
        help="Intermediate size of the feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim', 
        default=256, 
        type=int,                        
        help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument(
        '--dropout', 
        default=0.1, 
        type=float,                        
        help="Dropout applied in the transformer"
    )
    parser.add_argument(
        '--nheads', 
        default=8, 
        type=int, 
        help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument(
        '--num_queries_matching', 
        default=10, 
        type=int
    )
    parser.add_argument(
        '--num_queries_rec', 
        default=100, 
        type=int
    )
    parser.add_argument(
        '--pre_norm', 
        action='store_true'
    )
    parser.add_argument(
        '--fix_timestep', 
        default=None, 
        type=int, 
        help="use one timestep to train the model"
    )
    parser.add_argument(
        '--transformer_decoder_only', 
        action='store_true'
    )
    parser.add_argument(
        '--reset_optimizer', 
        action='store_true'
    )
    # rec
    parser.add_argument(
        "--combine_datasets_val", 
        nargs="+", 
        help="List of datasets to combine for eval", 
        default=['refcoco', 'refcoco+', 'refcocog']
    )
    parser.add_argument(
        "--no_freeze_text_encoder", 
        dest="freeze_text_encoder", 
        action="store_false", 
        help="Whether to freeze the weights of the text encoder"
    )
    parser.add_argument(
        '--text_encoder_type', 
        type=str, 
        default='roberta-base'
    )
    parser.add_argument(
        "--no_contrastive_align_loss", 
        dest="contrastive_align_loss", 
        action="store_false",
        help="Whether to add contrastive alignment loss"
    )
    parser.add_argument(
        "--contrastive_loss_hdim", 
        type=int, 
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss"
    )
    parser.add_argument(
        "--masks", 
        action="store_true"
    ) 
    # for segmentation
    parser.add_argument(
        "--no_detection", 
        action="store_true", 
        help="Whether to train the detector"
    )
    parser.add_argument(
        "--val_fullset", 
        dest='val_subset', 
        action="store_false"
    )
    parser.add_argument(
        "--dataset_name_val", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--dataset_name_matching", 
        type=str, 
        default='mscoco_hard_negative'
    )
    # Criterion
    parser.add_argument(
        "--set_loss", 
        default="hungarian", 
        type=str, 
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss"
    )
    parser.add_argument(
        "--temperature_NCE", 
        type=float, 
        default=0.07, 
        help="Temperature in the  temperature-scaled cross entropy loss"
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class", 
        default=1, 
        type=float,
        help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_bbox", 
        default=5, 
        type=float,
        help="L1 box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_giou", 
        default=2, 
        type=float,
        help="giou box coefficient in the matching cost"
    )
    # Loss coefficients
    parser.add_argument(
        "--ce_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--bbox_loss_coef", 
        default=5, 
        type=float
    )
    parser.add_argument(
        "--giou_loss_coef", 
        default=2, 
        type=float
    )
    parser.add_argument(
        "--contrastive_align_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--contrastive_i2t_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--contrastive_t2i_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--eos_coef", 
        default=0.1, 
        type=float,
        help="Relative classification weight of the no-object class"
    )
    parser.add_argument(
        "--unet_feature", 
        type=str, 
        required=True, 
        default=None
    )
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config, 
        kwargs_handlers=[ddp_kwargs]
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
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
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # unet.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    register_unet_output(unet)
    dpt, criterion, loss_weight_dict = build_model_combined(args, unet)
    contrastive_loss = ContrastiveLossClass()
    dpt.to(accelerator.device)
    criterion.to(accelerator.device)
    contrastive_loss.to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            dpt.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        # lora_layers.parameters(),
        dpt.parameters(), 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process:
        n_parameters = sum(p.numel() for p in dpt.parameters() if p.requires_grad)
        print('>> number of params:{:.2f}M'.format(n_parameters / 1e6))

    # ############################################# dataset #############################################
    train_dataset = get_dataset(args.dataset_name, args.img_root, args=args, split='train', combined=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=partial(util.misc.collate_fn_combined, False),
                                                num_workers=args.dataloader_num_workers, shuffle=True, pin_memory=True)
    # Val set for ReC
    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list", "split"])
    val_tuples = []
    args.combine_datasets_val = [args.dataset_name_val] if len(args.combine_datasets_val) == 0 else args.combine_datasets_val
    skip_margins = {'refcoco': 8, 'refcoco+': 16, 'refcocog': 8}
    for dset_name in args.combine_datasets_val:
        val_dataset = get_dataset(dset_name, args.img_root, args=args, split='val', combined=False)
        if args.val_subset:
            val_dataset = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset)))[::skip_margins[dset_name]])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, collate_fn=partial(util.misc.collate_fn, False),
                                                    num_workers=args.dataloader_num_workers, shuffle=False, pin_memory=True)
        val_dataloader = accelerator.prepare(val_dataloader)
        base_ds = get_coco_api_from_dataset(val_dataset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=val_dataloader, base_ds=base_ds, evaluator_list=None, split='val'))
    # Val set for Matching
    val_dataset = get_dataset('val_' + args.dataset_name_matching, args.img_root, transform=None, split='val', neg_img=False, hard_neg=True)
    val_dataset2 = get_dataset('val_' + args.dataset_name_matching, args.img_root, transform=None, split='val', neg_img=True, hard_neg=False)
    if args.val_subset:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset)))[::20])
        val_dataset2 = torch.utils.data.Subset(val_dataset2, list(range(len(val_dataset2)))[::20])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, 
                                                    num_workers=args.dataloader_num_workers // 2)
    val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=args.val_batch_size, shuffle=False, 
                                                    num_workers=args.dataloader_num_workers // 2)
    val_dataloader, val_dataloader2 = accelerator.prepare(val_dataloader, val_dataloader2)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.reset_optimizer:
        dpt, criterion, contrastive_loss, train_dataloader = accelerator.prepare(
            dpt, criterion, contrastive_loss, train_dataloader)
    else:
        dpt, criterion, contrastive_loss, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        dpt, criterion, contrastive_loss, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("DPT", config=vars(args), 
                        init_kwargs={"wandb":{"name": f"{args.run_name}"}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if args.resume_from_checkpoint != "latest" and args.output_dir != os.path.dirname(args.resume_from_checkpoint):
                state_path = args.resume_from_checkpoint
            else:
                state_path = os.path.join(args.output_dir, path)
            accelerator.load_state(state_path, strict=False)
            report_inconsistent_ckpt(accelerator, dpt, state_path, logger)
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            accelerator.print(f'resume_step = {resume_step}')

    if args.reset_optimizer:
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
        logger.info('NOTE: Reset the optimizer and the lr_scheduler!')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    
    for epoch in range(first_epoch, args.num_train_epochs):
        # unet.train()
        dpt.train()
        criterion.train()
        losses_to_show = ['loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error', 'loss_contrastive_align', 
                            'loss_contrastive_i2t', 'loss_contrastive_t2i', 'loss']
        train_loss_gathered = {k: 0.0 for k in losses_to_show}
        for step, batch_dict in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(dpt):
                device = accelerator.device
                samples = batch_dict["samples"].to(device)
                positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
                targets = batch_dict["targets"]
                txts_neg = torch.stack([neg['text_neg'] for neg in batch_dict["negatives"]])
                txts_rand = torch.stack([neg['text_rand'] for neg in batch_dict["negatives"]])
                # imgs, mask = samples.decompose()
                imgs, imgs_neg, imgs_rand = None, None, None
                imgs_latent, imgs_neg_latent, imgs_rand_latent = None, None, None
                if 'img_latent' in targets[0]:
                    imgs_latent = torch.stack([t['img_latent'] for t in targets]).to(device)
                    imgs_neg_latent = torch.stack([neg['img_neg_latent'] for neg in batch_dict["negatives"]]).to(device)
                    imgs_rand_latent = torch.stack([neg['img_rand_latent'] for neg in batch_dict["negatives"]]).to(device)
                else:
                    imgs, mask = samples.decompose()
                    imgs_neg = torch.stack([neg['img_neg'] for neg in batch_dict["negatives"]])
                    imgs_rand = torch.stack([neg['img_rand'] for neg in batch_dict["negatives"]])

                bsz = txts_neg.shape[0]
                # Convert images to latent space
                noisy_latents, timesteps, noise, target_noise = get_noisy_latents(args, imgs, vae, noise_scheduler, weight_dtype, latents=imgs_latent)
                # Get the text embedding for conditioning
                text_for_object = [t["caption"] for t in targets]
                text_for_object_ids = tokenizer(text_for_object, max_length=tokenizer.model_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids
                encoder_hidden_states = text_encoder(text_for_object_ids.to(device))
                encoder_hidden_states_neg = text_encoder(txts_neg)
                encoder_hidden_states_rand = text_encoder(txts_rand)
                text_uncond = tokenizer([''], max_length=tokenizer.model_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids
                encoder_hidden_states_uncond = text_encoder(text_uncond.to(device))[0]
                encoder_hidden_states_uncond = encoder_hidden_states_uncond.expand(bsz, -1, -1)
                text_whole_img = [t["caption_whole_image"] for t in targets]
                text_whole_img_ids = tokenizer(text_whole_img, max_length=tokenizer.model_max_length, padding="max_length", 
                                        truncation=True, return_tensors="pt").input_ids
                encoder_hidden_states_whole_img = text_encoder(text_whole_img_ids.to(device))
                # forward positive
                outputs = dpt(noisy_latents, timesteps, encoder_hidden_states[0], text_for_object, return_unet_feat=args.unet_feature) 
                # forward negative texts
                dict_pos = dpt(noisy_latents, timesteps, encoder_hidden_states_whole_img[0], only_matching=True, return_unet_feat=args.unet_feature, return_dict=True) 
                q_emb_i_t, logit_scale = dict_pos['q_emb_matching'], dict_pos['logit_scale']
                q_emb_i_tneg, _ = dpt(noisy_latents, timesteps, encoder_hidden_states_neg[0], only_matching=True, return_unet_feat=args.unet_feature)
                q_emb_i_trand, _ = dpt(noisy_latents, timesteps, encoder_hidden_states_rand[0], only_matching=True, return_unet_feat=args.unet_feature)
                # forward negative images
                noisy_latents_neg, _, _, target_neg = get_noisy_latents(args, imgs_neg, vae, noise_scheduler, weight_dtype, noise_and_t=(noise, timesteps), latents=imgs_neg_latent)
                noisy_latents_rand, _, _, target_rand = get_noisy_latents(args, imgs_rand, vae, noise_scheduler, weight_dtype, noise_and_t=(noise, timesteps), latents=imgs_rand_latent)
                q_emb_ineg_t, _ = dpt(noisy_latents_neg, timesteps, encoder_hidden_states_whole_img[0], only_matching=True, return_unet_feat=args.unet_feature)
                q_emb_irand_t, _ = dpt(noisy_latents_rand, timesteps, encoder_hidden_states_whole_img[0], only_matching=True, return_unet_feat=args.unet_feature)

                img_cond_different_txt = torch.stack([q_emb_i_t[0, :, :, :], q_emb_i_tneg[0, :, :, :], 
                                                    q_emb_i_trand[0, :, :, :]], dim=1) # (bs, 3, n_query_m, dim)
                different_img_cond_same_txt = torch.stack([q_emb_i_t[0, :, :, :], q_emb_ineg_t[0, :, :, :], 
                                                    q_emb_irand_t[0, :, :, :]], dim=1) # (bs, 3, n_query_m, dim)
                all_txt = torch.stack([encoder_hidden_states_whole_img.pooler_output, 
                            encoder_hidden_states_neg.pooler_output, 
                            encoder_hidden_states_rand.pooler_output], dim=1)

                loss_contrast_t2i, loss_contrast_i2t = contrastive_loss(img_cond_different_txt, different_img_cond_same_txt, all_txt, logit_scale)
                loss_dict = {'loss_contrastive_i2t': loss_contrast_i2t, 'loss_contrastive_t2i': loss_contrast_t2i}
                loss_dict.update(criterion(outputs, targets, positive_map))
                loss = sum(loss_dict[k] * loss_weight_dict[k] for k in loss_dict.keys() if k in loss_weight_dict)
                loss_dict['loss'] = loss

                # Gather the losses across all processes for logging (if we use distributed training).
                def gather_loss(loss):
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    return avg_loss.item() / args.gradient_accumulation_steps

                for k, v in train_loss_gathered.items():
                    train_loss_gathered[k] += gather_loss(loss_dict[k])

                # Backpropagate
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = dpt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                for k, v in train_loss_gathered.items():
                    accelerator.log({k: v}, step=global_step)
                train_loss_gathered = {k: 0.0 for k in losses_to_show}

                if (global_step % args.checkpointing_steps == 0 or global_step in [10, 20, 50, 100, 150, 200, 300, 400, 600]):
                    if accelerator.is_main_process and 'debug' not in args.run_name:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        if args.validation_prompts is not None and global_step % args.checkpointing_steps == 0:
                            all_prompts_str = '\n'.join(args.validation_prompts)
                            logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images for {len(args.validation_prompts)} prompts:\n"
                            f"{all_prompts_str}")
                            generate(args, accelerator, unet, epoch, weight_dtype, use_lora=False)
                    # evaluation
                    dpt.eval()
                    models = (dpt, vae, noise_scheduler, tokenizer, text_encoder)
                    # - rec
                    test_stats = eval_rec(args, models, criterion, val_tuples, loss_weight_dict, device, weight_dtype)
                    # - matching
                    models = (dpt, vae, text_encoder, tokenizer)
                    txt_acc, img_acc, txt_max_more_than_onces, img_max_more_than_onces = eval_acc(args, val_dataloader, val_dataloader2, 
                                                                        models, noise_scheduler, encoder_hidden_states_uncond, 
                                                                        score_batch_func=partial(score_batch_func, return_unet_feat=args.unet_feature))
                    def gather_metrics(x):
                        x = torch.as_tensor(x, device=device)
                        x = accelerator.gather(x)
                        return x.float().mean().item()

                    txt_acc, img_acc, txt_max_more_than_onces, img_max_more_than_onces = \
                            (gather_metrics(x) for x in [txt_acc, img_acc, txt_max_more_than_onces, img_max_more_than_onces]) 

                    if accelerator.is_main_process:
                        test_stats.update({'MSCOCO Val Accuracy Txt': txt_acc, 'Max more than once': txt_max_more_than_onces, 
                                        'MSCOCO Val Accuracy Img': img_acc, 'Max more than once': img_max_more_than_onces, 
                                        'Overall Val Accuracy': (txt_acc + img_acc) / 2})
                        for dataset_name_val in args.combine_datasets_val:
                            test_stats[f'{dataset_name_val}_{dataset_name_val}_k1'] = \
                                            test_stats[f'{dataset_name_val}_{dataset_name_val}'][0]
                        logger.info(test_stats)
                        test_stats['global_step'] = global_step
                        accelerator.log(test_stats)

                    torch.cuda.empty_cache()
                    dpt.train()

            loss_name_abbr = {'loss_ce': 'ce', 'loss_bbox': 'bbox', 'loss_giou': 'giou', 'cardinality_error': 'cardi', 
                            'loss_contrastive_align': 'align', 'loss_contrastive_i2t': 'i2t', 'loss_contrastive_t2i': 't2i', 
                            'loss': 'l'}
            logs = {loss_name_abbr[k]: loss_dict[k].detach().item() for k in losses_to_show}
            logs.update({'loss': loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break

    # Final inference
    # Load previous pipeline
    if accelerator.is_main_process:
        generate(args, accelerator, unet, epoch, weight_dtype, use_lora=False)
    accelerator.end_training()


if __name__ == "__main__":
    main()