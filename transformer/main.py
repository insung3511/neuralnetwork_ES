import torch.nn as nn
import torch

from save_png import save_reward_plot
from save_gif import save_animation, save_animation_sequence, save_animation_3d_surface
import os

from modelTransformer import TransformerModule, set_seed, validate_model_output
from es import EvolutionStrategies
from dataRandom import get_dataloader

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
RANDOM_SEED = 42

generation = 100
mean_rewards = []

set_seed(RANDOM_SEED)
dataloader = get_dataloader(batch_size=32, num_samples=1000, sequence_length=10, feature_dim=64, dataloader_shuffle=True)
# model expects (batch, seq_len, input_dim) and we set input_dim=feature_dim
model = TransformerModule(sequence_length=10, input_dim=64, model_dim=64, num_heads=4, num_layers=2, ff_dim=256, output_dim=1).to(DEVICE)
es = EvolutionStrategies(model, sigma=0.1, learning_rate=0.2, population_size=60, device=DEVICE)
criterion = nn.MSELoss()

def get_reward(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x)
        loss = criterion(preds, y)
    reward = -float(loss.item())
    if not torch.isfinite(torch.tensor(reward)):
        return -1e9
    return reward
print(f"[INFO] Initialzation complete.")

# Validate model output shape
for batch_x, batch_y in dataloader:
    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
    assert validate_model_output(model, batch_x, batch_y)
    break
print(f"[INFO] Model output shape validated.")

print(f"[INFO] Starting training for {generation + 1} generations.")
model_preds = []
candidates_preds = []

num_candidates = 6

for gen in range(generation):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        mean_reward = es.train_step(lambda m: get_reward(m, batch_x, batch_y))
    if not torch.isfinite(torch.tensor(mean_reward)):
        print(f"[ERR-] Generation {gen}: mean reward is not finite ({mean_reward}). Stopping.")
        break
    mean_rewards.append(mean_reward)
    print(f"[RUN-] Generation {gen}: Mean Reward: {mean_reward}")

    model.eval()
    with torch.no_grad():
        preds = model(batch_x)  # (batch, seq_len, 1)
    # store 1D prediction for sample 0 (seq_len,)
    model_preds.append(preds[0, :, 0].cpu().numpy())

    base = es.get_flat_params().detach().clone()
    candidate_preds_gen = []
    for _ in range(num_candidates):
        noise = torch.randn_like(base)
        es.set_flat_params(base + es.sigma * noise)
        model.eval()
        with torch.no_grad():
            candidate_pred = model(batch_x)
        # store 1D prediction for sample 0
        candidate_preds_gen.append(candidate_pred[0, :, 0].cpu().numpy())
    es.set_flat_params(base)
    candidates_preds.append(candidate_preds_gen)

out_dir = os.path.dirname(__file__)
save_reward_plot(os.path.join(out_dir, 'transformer_reward_plot.png'), mean_rewards)
# Save sequence animation for sample 0
save_animation_sequence(out_dir, batch_y[0, :, 0].cpu().numpy(), model_preds, candidates_preds, mean_rewards)
# Save 3D surface animation
save_animation_3d_surface(out_dir, batch_y[0, :, 0].cpu().numpy(), model_preds, mean_rewards)

print("Training and visualization complete.")