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
