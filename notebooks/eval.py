import diffusers
import torch
import sys
sys.path.append("../")
from DDPMPipelineDropout import DDPMPipeline
import utils

from argparse import ArgumentParser

parse = ArgumentParser()
parse.add_argument("-m", "--model", type=str)
parse.add_argument("-s", "--start_range", type=int)
parse.add_argument("-e", "--end_range", type=int)
parse.add_argument("--samples", type=int)
parse.add_argument("-b", "--batch_size", type=int)
parse.add_argument("-n", "--num_images", type=int)
args = parse.parse_args()

model_path = args.model
model, config = utils.load_model(model_path)
scheduler = diffusers.DDPMScheduler(num_train_timesteps=config.num_train_timestamps)
pipeline = DDPMPipeline(unet=model, scheduler=scheduler).to("cuda")

#Set Eval params here
config.bayesian_avg_range = (args.start_range, args.end_range)
config.bayesian_avg_samples = args.samples
print(config.__dict__)

fid_score, inception_score = utils.calculate_metrics(config, pipeline, batch_size=args.batch_size, num_images=args.num_images)

try :
    results = torch.load("../results.dict")
except:
    results = {}

if config.run_name in results:
    results[config.run_name]["fid_score"] = fid_score.item()
    results[config.run_name]["path"] = model_path
    results[config.run_name]["inception_score_mean"] = inception_score[0].item()
    results[config.run_name]["inception_score_std"] = inception_score[1].item()
    results[config.run_name]["config"] = config.__dict__
else:
    results[config.run_name] = {}
    results[config.run_name]["fid_score"] = fid_score.item()
    results[config.run_name]["path"] = model_path
    results[config.run_name]["inception_score_mean"] = inception_score[0].item()
    results[config.run_name]["inception_score_std"] = inception_score[1].item()
    results[config.run_name]["config"] = config.__dict__

torch.save(results, "../results.dict")