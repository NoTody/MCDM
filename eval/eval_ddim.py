import diffusers
import torch
import sys
sys.path.append("../")
from DDPMPipelineDropout import DDPMPipeline
from DDIMPipelineDropout import DDIMPipeline
import utils

from argparse import ArgumentParser

parse = ArgumentParser()
parse.add_argument("-m", "--model", type=str)
parse.add_argument("-s", "--start_range", type=int)
parse.add_argument("-e", "--end_range", type=int)
parse.add_argument("--samples", type=int)
parse.add_argument("-b", "--batch_size", type=int)
parse.add_argument("-n", "--num_images", type=int)
parse.add_argument("--name", type=str)
parse.add_argument("--stats", action="store_true")
args = parse.parse_args()

model_path = args.model
model, config = utils.load_model(model_path)
scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
pipeline = DDIMPipeline(unet=model, scheduler=scheduler).to("cuda")

#Set Eval params here
config.bayesian_avg_range = (args.start_range, args.end_range)
config.bayesian_avg_samples = args.samples
print(config.__dict__)

if args.stats:
    fid_score, inception_score, means, stds = utils.calculate_metrics(config, pipeline, batch_size=args.batch_size, num_images=args.num_images, generation_progress=True, calculate_stats=args.stats)
else:
    fid_score, inception_score = utils.calculate_metrics(config, pipeline, batch_size=args.batch_size, num_images=args.num_images, generation_progress=True)

try :
    results = torch.load("../results_ddim_lsun.dict")
except:
    results = {}

if args.name in results:
    results[args.name]["fid_score"] = fid_score.item()
    results[args.name]["path"] = model_path
    results[args.name]["inception_score_mean"] = inception_score[0].item()
    results[args.name]["inception_score_std"] = inception_score[1].item()
    if args.stats:
        results[args.name]["diffusion_mean"] = means
        results[args.name]["diffusion_std"] = stds 
    results[args.name]["config"] = config.__dict__
else:
    results[args.name] = {}
    results[args.name]["fid_score"] = fid_score.item()
    results[args.name]["path"] = model_path
    results[args.name]["inception_score_mean"] = inception_score[0].item()
    results[args.name]["inception_score_std"] = inception_score[1].item()
    if args.stats:
        results[args.name]["diffusion_mean"] = means
        results[args.name]["diffusion_std"] = stds 
    results[args.name]["config"] = config.__dict__

torch.save(results, "../results_ddim_lsun.dict")
