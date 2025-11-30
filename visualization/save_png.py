import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_reward_plot(out_path: str, mean_rewards):
    """Save a static mean-reward plot to `out_path`.

    `mean_rewards` should be an iterable of floats.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(mean_rewards, label='mean reward')
    plt.xlabel('Generation')
    plt.ylabel('Mean reward')
    plt.title('ES training progress')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
