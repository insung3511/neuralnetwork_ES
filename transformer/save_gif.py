import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation


def save_animation(out_dir: str, x, y, model_preds, candidates_preds, mean_rewards, fps: int = 8):
    """Create and save an animation (GIF or MP4) showing predictions over generations.

    - `x`, `y` may be numpy arrays or torch tensors (function will convert to numpy).
    - `model_preds` is a list of 1D numpy arrays (one per frame).
    - `candidates_preds` is a list (frames) of lists (candidates) of 1D arrays.
    - `mean_rewards` is a list of scalar rewards used for titles.
    """
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, 'es_training.gif')
    mp4_path = os.path.join(out_dir, 'es_training.mp4')

    # ensure numpy arrays
    try:
        x_np = np.asarray(x.cpu()) if hasattr(x, 'cpu') else np.asarray(x)
    except Exception:
        x_np = np.asarray(x)
    try:
        y_np = np.asarray(y.cpu()) if hasattr(y, 'cpu') else np.asarray(y)
    except Exception:
        y_np = np.asarray(y)

    x_flat = x_np.squeeze()
    y_flat = y_np.squeeze()

    num_candidates = len(candidates_preds[0]) if candidates_preds else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_flat, y_flat, color='k', lw=2, label='target')
    main_line, = ax.plot([], [], color='tab:blue', lw=2, label='model')
    cand_lines = [ax.plot([], [], color='tab:orange', alpha=0.6, lw=1)[0] for _ in range(num_candidates)]
    title = ax.text(0.5, 1.03, '', transform=ax.transAxes, ha='center')
    ax.set_xlim(float(x_flat.min()), float(x_flat.max()))
    # compute y limits robustly using percentiles across target and predictions
    arrays = [y_flat]
    try:
        for p in model_preds:
            arrays.append(np.asarray(p).ravel())
    except Exception:
        pass
    try:
        for frame in candidates_preds:
            for c in frame:
                arrays.append(np.asarray(c).ravel())
    except Exception:
        pass

    if arrays:
        stacked = np.concatenate([a.ravel() for a in arrays if a is not None and a.size > 0])
        # use 1st and 99th percentile to ignore extreme outliers
        try:
            p1, p99 = np.nanpercentile(stacked, [1.0, 99.0])
            if not np.isfinite(p1) or not np.isfinite(p99):
                raise ValueError('non-finite percentiles')
        except Exception:
            p1, p99 = float(y_flat.min()), float(y_flat.max())
        if p1 == p99:
            # expand a bit if flat
            p1 -= 0.5
            p99 += 0.5
        margin = max(0.1 * (p99 - p1), 0.5)
        ax.set_ylim(p1 - margin, p99 + margin)
    else:
        ax.set_ylim(float(y_flat.min()) - 0.5, float(y_flat.max()) + 0.5)
    ax.grid(True)
    ax.legend(loc='upper left')

    def init():
        main_line.set_data([], [])
        for line in cand_lines:
            line.set_data([], [])
        title.set_text('')
        return [main_line, *cand_lines, title]

    def update(frame):
        main_line.set_data(x_flat, model_preds[frame])
        for i, line in enumerate(cand_lines):
            line.set_data(x_flat, candidates_preds[frame][i])
        title.set_text(f'Generation {frame+1} — mean reward {mean_rewards[frame]:.4f}')
        return [main_line, *cand_lines, title]

    anim = animation.FuncAnimation(fig, update, frames=len(model_preds), init_func=init, blit=True)

    # try save GIF via Pillow, else MP4 via ffmpeg
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        return {'gif': gif_path}
    except Exception as e:
        try:
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps, metadata=dict(artist='es-demo'))
            anim.save(mp4_path, writer=writer)
            plt.close(fig)
            return {'mp4': mp4_path}
        except Exception as e2:
            plt.close(fig)
            raise RuntimeError(f'Failed to save animation: {e}, {e2}')


def save_animation_3d_surface(out_dir: str, y_target, model_preds, mean_rewards, fps: int = 8):
    """Save a 3D surface animation showing model predictions vs ground truth across generations and timesteps.

    - `y_target` should be a 1D array (seq_len,) containing the ground-truth values.
    - `model_preds` is a list of 1D arrays (one per generation).
    - `mean_rewards` is a list of scalars for frame titles.
    
    Creates a 3D surface plot: x=timestep, y=generation, z=prediction value.
    Shows both target (reference, constant) and model predictions (evolving).
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, '3d_surface_evolution.gif')
    mp4_path = os.path.join(out_dir, '3d_surface_evolution.mp4')

    # ensure numpy arrays
    try:
        y_np = np.asarray(y_target.cpu()) if hasattr(y_target, 'cpu') else np.asarray(y_target)
    except Exception:
        y_np = np.asarray(y_target)

    seq_len = int(np.asarray(model_preds[0]).ravel().shape[0]) if model_preds else y_np.size
    num_gens = len(model_preds)

    # Build matrix: (num_gens, seq_len) where each row is a generation's predictions
    pred_matrix = np.array([np.asarray(p).ravel()[:seq_len] for p in model_preds])

    # Create target surface: repeat target for each generation (constant reference)
    target_matrix = np.tile(y_np.squeeze(), (num_gens, 1))

    # Create meshgrid for x (timestep) and y (generation)
    X = np.arange(seq_len)
    Y = np.arange(num_gens)
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    # Compute Z limits for consistent scaling
    all_vals = np.concatenate([pred_matrix.ravel(), target_matrix.ravel()])
    z_min, z_max = np.nanpercentile(all_vals, [1, 99])
    z_margin = max(0.1 * (z_max - z_min), 0.5)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    title_text = fig.suptitle('', fontsize=12)

    def update(frame):
        ax.clear()
        
        # Plot target surface (reference, light semi-transparent)
        Z_target = target_matrix[:frame+1, :]
        X_curr = X_mesh[:frame+1, :]
        Y_curr = Y_mesh[:frame+1, :]

        surf_target = ax.plot_surface(X_curr, Y_curr, Z_target, cmap='Reds', alpha=0.8, label='Target (ground truth)')
        
        # Plot prediction surface (main, vibrant)
        Z_pred = pred_matrix[:frame+1, :]
        surf_pred = ax.plot_surface(X_curr, Y_curr, Z_pred, cmap='viridis', alpha=0.8, label='Model predictions')
        
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Generation', fontsize=10)
        ax.set_zlabel('Prediction Value', fontsize=10)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_title(f'3D Prediction Surface vs Target | Generation {frame+1} — Mean Reward: {mean_rewards[frame]:.4f}', fontsize=11)
        
        # Fixed viewing angle for clarity
        ax.view_init(elev=20, azim=45)
        
        return [surf_target, surf_pred]

    anim = animation.FuncAnimation(fig, update, frames=num_gens, blit=False)

    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        return {'gif': gif_path}
    except Exception as e:
        try:
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps, metadata=dict(artist='es-demo'))
            anim.save(mp4_path, writer=writer)
            plt.close(fig)
            return {'mp4': mp4_path}
        except Exception as e2:
            plt.close(fig)
            raise RuntimeError(f'Failed to save 3D surface animation: {e}, {e2}')


def save_animation_sequence(out_dir: str, y_target, model_preds, candidates_preds, mean_rewards, fps: int = 8, sample_idx: int = 0):
    """Save a 2D per-timestep sequence animation for a single sample.

    - `y_target` should be a 1D array (seq_len,) containing the ground-truth values for the sample.
    - `model_preds` is a list of 1D arrays (one per generation/frame).
    - `candidates_preds` is a list (frames) of lists (candidates) of 1D arrays.
    - `mean_rewards` is a list of scalars for frame titles.
    """
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, 'sequence_evolution.gif')
    mp4_path = os.path.join(out_dir, 'sequence_evolution.mp4')

    # ensure numpy arrays
    try:
        y_np = np.asarray(y_target.cpu()) if hasattr(y_target, 'cpu') else np.asarray(y_target)
    except Exception:
        y_np = np.asarray(y_target)

    seq_len = int(np.asarray(model_preds[0]).ravel().shape[0]) if model_preds else y_np.size
    x_axis = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_axis, y_np.squeeze(), color='k', lw=2, label='target')
    main_line, = ax.plot([], [], color='tab:blue', lw=2, label='model')
    num_candidates = len(candidates_preds[0]) if candidates_preds else 0
    cand_lines = [ax.plot([], [], color='tab:orange', alpha=0.6, lw=1)[0] for _ in range(num_candidates)]
    title = ax.text(0.5, 1.03, '', transform=ax.transAxes, ha='center')

    # Determine y-limits robustly
    arrays = [y_np.ravel()]
    try:
        for p in model_preds:
            arrays.append(np.asarray(p).ravel())
    except Exception:
        pass
    try:
        for frame in candidates_preds:
            for c in frame:
                arrays.append(np.asarray(c).ravel())
    except Exception:
        pass

    if arrays:
        stacked = np.concatenate([a.ravel() for a in arrays if a is not None and a.size > 0])
        try:
            p1, p99 = np.nanpercentile(stacked, [1.0, 99.0])
            if not np.isfinite(p1) or not np.isfinite(p99):
                raise ValueError('non-finite percentiles')
        except Exception:
            p1, p99 = float(y_np.min()), float(y_np.max())
        if p1 == p99:
            p1 -= 0.5
            p99 += 0.5
        margin = max(0.1 * (p99 - p1), 0.5)
        ax.set_ylim(p1 - margin, p99 + margin)
    else:
        ax.set_ylim(float(y_np.min()) - 0.5, float(y_np.max()) + 0.5)

    ax.set_xlim(0, seq_len - 1)
    ax.grid(True)
    ax.legend(loc='upper left')

    def init():
        main_line.set_data([], [])
        for line in cand_lines:
            line.set_data([], [])
        title.set_text('')
        return [main_line, *cand_lines, title]

    def update(frame):
        main_line.set_data(x_axis, model_preds[frame])
        for i, line in enumerate(cand_lines):
            line.set_data(x_axis, candidates_preds[frame][i])
        title.set_text(f'Generation {frame+1} — mean reward {mean_rewards[frame]:.4f}')
        return [main_line, *cand_lines, title]

    anim = animation.FuncAnimation(fig, update, frames=len(model_preds), init_func=init, blit=True)

    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        return {'gif': gif_path}
    except Exception as e:
        try:
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps, metadata=dict(artist='es-demo'))
            anim.save(mp4_path, writer=writer)
            plt.close(fig)
            return {'mp4': mp4_path}
        except Exception as e2:
            plt.close(fig)
            raise RuntimeError(f'Failed to save animation: {e}, {e2}')
