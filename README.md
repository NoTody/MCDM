# BMLProject
Monte Carlo Diffusion Models

# Dependencies
Chekc and install all dependencies
```
pip install requirements.txt 
```
accelerate==0.14.0, torch==1.8.1, diffusers==0.7.2

# Run Examples
## Train
Changing hyperparameters in "config_dict" dictionary of corresponding files and with Accelerate from https://huggingface.co/docs/accelerate/index installed 
Run DDIM train:
```
accelerate launch ddim_train.py
```
Run DDPM train:
```
accelerate launch ddpm_train.py
```
## Evaluation
Run DDIM evlauation:
```
python ./notebooks/eval_ddim.py --model <model_path> --start_range 0 --end_range 1000 --s    amples 1 --batch_size 128 --num_images 1000 --name <dict_name>
```
Run DDPM evlauation:
```
python eval_ddpm.py --model <model_path> --start_range 0 --end_range 250 --sam    ples 5 --batch_size 128 --num_images 1000 --name <dict_name>
```
