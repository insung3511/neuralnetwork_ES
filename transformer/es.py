# Evolution Strategies (ES) for optimizing neural networks in PyTorch
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

# Evolution Strategies class
class EvolutionStrategies:
    """Implements the Evolution Strategies optimization algorithm for neural networks.
    Example usage:
        get_reward = lambda model: ... # Define a function that returns the reward for the given model
        model = SomeNeuralNetworkModel()
        es = EvolutionStrategies(model, sigma=0.1, learning_rate=0.01
        for generation in range(num_generations):
            es.train_step(get_reward)
    """
    def __init__(self, model, sigma=0.1, learning_rate=0.01, population_size=50, device='cpu'):
        """Initialize the ES optimizer.
        Args:
            model (nn.Module): The neural network model to optimize.
            sigma (float): Standard deviation of the noise.
            learning_rate (float): Learning rate for parameter updates.
            population_size (int): Number of perturbations to sample.
            device (str): The device to use for all computations. The model should be moved to this device before creating the optimizer.
        """
        self.model = model
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.population_size = population_size
        self.device = device

    def get_flat_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def set_flat_params(self, flat_params):
        prev_ind = 0
        for param in self.model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def get_noisy_params(self, noise):
        flat_params = self.get_flat_params()
        return flat_params + self.sigma * noise

    def train_step(self, get_reward):
        noise_population = []
        rewards = []
        # Use the current parameters as the base for all sampled perturbations.
        # Without this, successive perturbations are applied on top of the last
        # noisy parameters which causes parameters to grow uncontrollably.
        base_params = self.get_flat_params().detach().clone()

        for _ in range(self.population_size):
            # sample noise using the base params shape and device to stay consistent
            noise = torch.randn_like(base_params)
            noisy_params = base_params + self.sigma * noise
            self.set_flat_params(noisy_params)
            reward = get_reward(self.model)
            noise_population.append(noise)
            rewards.append(reward)

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        gradient = torch.zeros_like(self.get_flat_params())
        for noise, reward in zip(noise_population, normalized_rewards):
            gradient += reward * noise
        gradient /= self.population_size
        
        # Clip gradient to prevent explosion
        gradient = torch.clamp(gradient, -1.0, 1.0)

        flat_params = self.get_flat_params()
        updated_params = flat_params + (self.learning_rate / self.sigma) * gradient
        self.set_flat_params(updated_params)
        return rewards.mean().item()

def get_reward(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(x)
        loss = criterion(preds, y)
    reward = -float(loss.item())
    if not np.isfinite(reward):
        return -1e9
    return reward