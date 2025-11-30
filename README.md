# Evolution Strategies Demo

This repository/project works with AI Assistant (Copliot)

**Purpose:** A tiny demonstration of Evolution Strategies (ES) applied to simple neural networks (PyTorch). This repo provides runnable examples that show how ES explores parameter space, updates model weights, and how to visualize the training process.

**Repository layout (relevant files):**
- `visualization/main.py`: Run the ES demo and produce an animation showing the model and sampled candidate predictions per generation.
- `visualization/es.py`: ES optimizer implementation used to perturb and update model parameters.
- `visualization/model.py`: Small MLP and dataset utilities.
- `visualization/save_png.py`: Helper to save the mean-reward PNG.
- `visualization/save_gif.py`: Helper to save the animated GIF/MP4 of predictions.
- `visualization/validate.py`: Simple validation runner that trains on a train split and evaluates on held-out data and writes `validation_result.json`.

**Quickstart**

1. Install dependencies (recommended in a virtual environment):

```bash
cd /path/to/NeuralNetwork_ES
python3 -m venv .venv        # optional
source .venv/bin/activate    # optional
python3 -m pip install --user -r requirements.txt
```

2. Run the visualization demo (saves `es_training.png` and `es_training.gif`):

```bash
python3 visualization/main.py
```

3. Run the validation runner to get a simple validation MSE and parameter norm:

```bash
python3 visualization/validate.py
# writes visualization/validation_result.json
```

**Interpreting the results**

- **Reward meaning:** In the demo the reward is defined as the negative mean-squared error (MSE) between the model predictions and the target: `reward = -MSE`. ES is a black-box optimizer that *maximizes* reward, so increasing reward corresponds to decreasing the MSE.
- **Why L2 / MSE loss:** L2 loss (MSE) is the standard choice for regression problems because it is smooth, differentiable, and places larger penalty on large errors which often speeds learning toward reducing large deviations. In the ES demo we use negative MSE as the scalar reward since ES needs a scalar objective to maximize.
- **Limitations of L2:** MSE is sensitive to outliers (large errors). If you observe plots where the target looks flat (y-range very large), this usually means some model predictions or candidate samples are extreme outliers and have stretched the axis — the visualization includes robust percentile-based limits to mitigate that, but you should also investigate training stability.
- **Parameter norm & instability:** If parameter norms grow very large or rewards become NaN/Inf, try reducing `learning_rate` and/or `sigma`, use smaller `population_size`, or add clipping/regularization. The repository includes a `visualization/validate.py` helper to gather a final `param_norm` and validation MSE to help debugging.

**Practical tips (to improve ES training stability)**

- **Lower `sigma` (noise)**: Smaller perturbations reduce high-variance gradient estimates.
- **Lower `learning_rate`**: Smaller updates help avoid parameter explosion.
- **Increase `population_size`**: More samples reduce variance in the gradient estimate (at higher compute cost).
- **Use rank or centered ranking for rewards**: This reduces sensitivity to absolute reward scale and outliers (not implemented by default here but worth trying).
- **Clip parameter updates or outputs**: For visualization you can clip predictions; for real training you can clip gradients or parameter norms.

**References & further reading**

- Salimans, Tim, et al. "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." 2017. (OpenAI blog/paper) — https://arxiv.org/abs/1703.03864
- A short tutorial on ES and finite-difference gradient estimates — search for "evolution strategies tutorial" or check OpenAI's blog posts.
- PyTorch docs — https://pytorch.org/docs/stable/index.html
- Matplotlib animation docs — https://matplotlib.org/stable/api/animation_api.html

**If you'd like to extend this project**

- Add alternative loss functions (MAE, Huber) if you want robustness to outliers.
- Implement rank-based reward normalization inside `visualization/es.py` for more stable training.
- Add a Jupyter notebook that runs the demo and shows the animation inline.

If you want, I can add a short notebook and a few tuned hyperparameter presets for stable runs on CPU or MPS.


