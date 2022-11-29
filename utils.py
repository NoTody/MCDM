import torch
import torch.nn.functional as F
import numpy as np
from UnetDropout import UNet2DModel
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision
import diffusers
from DDPMPipelineDropout import DDPMPipeline
import math
from accelerate import Accelerator
from tqdm.auto import tqdm
from dataclasses import dataclass
import wandb


def un_normalize_image_tensor(image : torch.tensor):
    return ((image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]

class ffhq_Dataset(Dataset):
    def __init__(self, root_dir : str, transforms : torchvision.transforms):
        self.augmentation = transforms
        self.img_list = os.listdir(root_dir)
        self.img_list.remove("LICENSE.txt")
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.root_dir + self.img_list[idx])
        return self.augmentation(img)

def get_default_unet(config):
    model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
    down_dropout=config.down_dropout,
    mid_dropout=config.mid_dropout,
    up_dropout=config.up_dropout
    )
    return model

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def ddpm_evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
        bayesian_avg_samples=config.bayesian_avg_samples,
        bayesian_avg_range=config.bayesian_avg_range
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)
    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return image_grid

def save_model(model, config, epoch):
    torch.save({"config": config, "weights":model.state_dict}, config.output_dir+f"{config.run_name}-epoch-{epoch}.pth")

# train
def ddpm_train_loop(config : dataclass, model : torch.nn.Module, noise_scheduler : diffusers.DDPMScheduler, optimizer : torch.optim.Adam, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(config.run_name, config)
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if config.dataset == "CIFAR10":
                clean_images = batch[0]
            else:
                clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                eval_generations = ddpm_evaluate(config, epoch, pipeline)
                accelerator.log({"Generations" : wandb.Image(eval_generations)})

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                save_model(model, config, epoch)