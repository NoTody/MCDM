import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
import numpy as np
import torchvision
from torchvision import transforms
import diffusers
from diffusers import DDIMScheduler
#from diffusers import DDIMPipeline
from DDIMPipelineDropout import DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import sys
from PIL import Image
import utils
from utils import LSUNDataset


config_dict = {
    "run_name" : "DDIM-FFHQ-TEST_nodp",
    "num_inference_steps" : 100,
    "dataset" : "FFHQ", # "CIFAR10" / "FFHQ" / "LSUN_CHURCH"
    "image_size" : 128,  # the generated image resolution
    "train_batch_size" : 32,
    "eval_batch_size" : 16,  # how many images to sample during evaluation
    "num_epochs" : 300,
    "gradient_accumulation_steps" : 1,
    "learning_rate" : 2e-4,
    "lr_warmup_steps" : 300,
    "save_image_epochs" : 10,
    "save_model_epochs" : 10,
    "mixed_precision" : 'fp16',  # `no` for float32, `fp16` for automatic mixed precision
    "output_dir" : 'ddim-ffhq-nodp',  # the model namy locally and on the HF Hub
    "overwrite_output_dir" : True,  # overwrite the old model when re-running the notebook
    "seed" : 0,
    "down_dropout" : 0.0,
    "mid_dropout" : 0.0,
    "up_dropout" : 0.0,
    "bayesian_avg_samples" : 1,
    "bayesian_avg_range" : (0, 1000)
}

config = utils.get_config_class(config_dict)

# transforms
image_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Create Dataloader

if config.dataset == "FFHQ":
    # FFHQ dataset
    dataset = utils.ffhq_Dataset("./dataset/ffhq/thumbnails128x128/", image_transforms)
    config.image_size = 128
elif config.dataset == "CIFAR10":
    #cifar dataset
    dataset = torchvision.datasets.CIFAR10(root= "./dataset/", download=True, transform=image_transforms)
    config.image_size = 32
elif config.dataset == "LSUN_CHURCH":
    # lsun-church dataset
    dataset = LSUNDataset("./dataset/lsun-church/church_outdoor_train_lmdb_color_64.npy", transform=image_transforms)
    config.image_size = 64
else:
    raise ValueError("Invalid Dataset supplied")

train_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Setup for Training
model = utils.get_default_unet(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_loader) * config.num_epochs),
)

# noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timestamps)
noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
# Start Training
utils.ddim_train_loop(config, model, noise_scheduler, optimizer, train_loader, lr_scheduler)

