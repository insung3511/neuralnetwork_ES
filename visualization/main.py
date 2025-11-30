"""ES animation demo.

This script trains a tiny MLP using Evolution Strategies and records
the model's prediction after each generation. It also samples a few
noisy candidate models each generation to illustrate exploration.

Outputs:
 - `visualization/es_training.png` (mean reward curve)
 - `visualization/es_training.gif` (animation showing target, model,
   and sampled candidate predictions across generations)
"""

import os
import math
import torch
import torch.nn as nn

from es import EvolutionStrategies
from model import set_seed, SimpleMLP, make_dataset
from save_png import save_reward_plot
from save_gif import save_animation

set_seed(42)
device = 'mps'

x, y = make_dataset(100)
x, y = x.to(device), y.to(device)
print(f"Dataset shapes: x={x.shape}, y={y.shape}")

model = SimpleMLP().to(device)

# ES hyperparams
es = EvolutionStrategies(model, sigma=0.1, learning_rate=0.2, population_size=60, device=device)

criterion = nn.MSELoss()

def get_reward(model):
    model.eval()
    with torch.no_grad():
        preds = model(x)
        loss = criterion(preds, y)
    reward = -float(loss.item())
    if not math.isfinite(reward):
        return -1e9
    return reward

generations = 200
mean_rewards = []

# storage for animation
model_preds = []           # main model prediction per generation
candidates_preds = []      # list of lists: per generation, multiple candidate predictions

num_candidates = 6

for gen in range(1, generations + 1):
    mean_reward = es.train_step(get_reward)
    if not math.isfinite(mean_reward):
        print(f"Generation {gen}: mean reward is not finite ({mean_reward}). Stopping.")
        break
    mean_rewards.append(mean_reward)

    # evaluate current model prediction
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy().squeeze()
    model_preds.append(pred)

    # sample several noisy candidates (visualize exploration)
    base = es.get_flat_params().detach().clone()
    cand_list = []
    for _ in range(num_candidates):
        noise = torch.randn_like(base)
        noisy = base + es.sigma * noise
        es.set_flat_params(noisy)
        model.eval()
        with torch.no_grad():
            p = model(x).cpu().numpy().squeeze()
        cand_list.append(p)
    # restore base params
    es.set_flat_params(base)
    candidates_preds.append(cand_list)

    if gen % 10 == 0:
        print(f"Generation {gen:4d}  Mean reward: {mean_reward:.6f}")

# save static reward plot
out_dir = os.path.dirname(__file__)
png_path = os.path.join(out_dir, 'es_training.png')
save_reward_plot(png_path, mean_rewards)
print(f"Saved training plot to: {png_path}")

# create and save animation
res = save_animation(out_dir, x, y, model_preds, candidates_preds, mean_rewards, fps=8)
print('Saved animation result:', res)
