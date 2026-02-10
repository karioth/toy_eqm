import re
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_tf32_if_cuda(device):
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


class MLP(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, width), nn.GELU()]
            in_dim = width
        layers += [nn.Linear(in_dim, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# Change these if you change the model architecture.
MODEL_WIDTH = 128
MODEL_DEPTH = 4


def build_model(device):
    return MLP(width=MODEL_WIDTH, depth=MODEL_DEPTH).to(device)



def sample_spiral_x(B, device, turns=3.0, r_max=3.0):
    t = (2.0 * torch.pi * turns) * torch.rand(B, 1, device=device)
    r = (r_max / (2.0 * torch.pi * turns)) * t
    x = torch.cat([r * torch.cos(t), r * torch.sin(t)], dim=1)
    return x


def make_spiral_reference(device, n_ref=12000, turns=3.0, r_max=3.0):
    t = torch.linspace(0, 2.0 * torch.pi * turns, n_ref, device=device).unsqueeze(-1)
    r = (r_max / (2.0 * torch.pi * turns)) * t
    s = torch.cat([r * torch.cos(t), r * torch.sin(t)], dim=1)
    return s


@torch.no_grad()
def spiral_normalization(spiral_ref, eps=1e-6):
    mean = spiral_ref.mean(dim=0, keepdim=True)
    std = spiral_ref.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean


def sample_pair_batch(B, device, turns, r_max, mean, std):
    x = sample_spiral_x(B, device, turns=turns, r_max=r_max)
    x = normalize(x, mean, std)
    gamma = torch.rand(B, 1, device=device)
    eps1 = torch.randn(B, 2, device=device)
    eps2 = torch.randn(B, 2, device=device)
    z1 = gamma * x + (1.0 - gamma) * eps1
    z2 = gamma * x + (1.0 - gamma) * eps2
    return z1, z2, x, gamma


def field_and_prediction(model, z):
    f = model(z)
    xpred = z + f
    return f, xpred


def lp_norm(x, alpha=1.0, eps=1e-12):
    n = torch.linalg.norm(x, dim=-1).clamp_min(eps)
    if alpha == 1.0:
        return n
    return n.pow(alpha)


def attraction_weight(gamma, mode="none", bias=0.05, power=1.0):
    if bias < 0:
        raise ValueError("bias must be >= 0")
    g = gamma.squeeze(-1)
    d = 1.0 - g
    if mode == "inv":
        return 1.0 / (d + bias).pow(power)
    if mode == "none":
        return torch.ones_like(g)
    raise ValueError("reweight_mode must be 'inv' or 'none'")


@torch.no_grad()
def nearest_spiral_point(z, spiral_ref, chunk=4096):
    B = z.shape[0]
    best_d = torch.full((B,), float("inf"), device=z.device)
    best_idx = torch.zeros((B,), dtype=torch.long, device=z.device)
    base = 0
    for i in range(0, spiral_ref.shape[0], chunk):
        s = spiral_ref[i : i + chunk]
        d = torch.cdist(z, s)
        dmin, arg = d.min(dim=1)
        better = dmin < best_d
        best_d = torch.where(better, dmin, best_d)
        best_idx = torch.where(better, arg + base, best_idx)
        base += s.shape[0]
    return spiral_ref[best_idx], best_d.unsqueeze(-1)


@torch.no_grad()
def dist_to_spiral(z, spiral_ref):
    _, d = nearest_spiral_point(z, spiral_ref)
    return d


@torch.no_grad()
def mean_nn_distance(src, ref, chunk=2048):
    best = torch.full((src.shape[0],), float("inf"), device=src.device)
    for i in range(0, ref.shape[0], chunk):
        r = ref[i : i + chunk]
        d = torch.cdist(src, r)
        dmin, _ = d.min(dim=1)
        best = torch.minimum(best, dmin)
    return best.mean()


@torch.no_grad()
def symmetric_chamfer_distance(a, b, chunk=2048):
    d_ab = mean_nn_distance(a, b, chunk=chunk)
    d_ba = mean_nn_distance(b, a, chunk=chunk)
    return 0.5 * (d_ab + d_ba)


def plot_quiver_field(model, device, ax, spiral_ref,
                      lim=3.2, grid=25, scale=12.0,
                      mean=None, std=None):
    xs = torch.linspace(-lim, lim, grid, device=device)
    ys = torch.linspace(-lim, lim, grid, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    pts_in = normalize(pts, mean, std)
    V, _ = field_and_prediction(model, pts_in)
    V = V * std
    V = V.reshape(grid, grid, 2).detach().cpu()
    ax.clear()
    sp = spiral_ref.detach().cpu()
    ax.plot(sp[::4, 0].numpy(), sp[::4, 1].numpy(), ".", ms=0.5, alpha=0.4, c="steelblue")
    ax.quiver(X.cpu().numpy(), Y.cpu().numpy(),
              V[..., 0].numpy(), V[..., 1].numpy(),
              angles="xy", scale_units="xy", scale=scale, width=0.003, alpha=0.8)
    ax.set_title("f(z) displacement field")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")


def plot_pred_scatter(model, device, ax, spiral_ref, B=4000, turns=3.0,
                      r_max=3.0, mean=None, std=None):
    z1, _, _, gamma = sample_pair_batch(
        B, device, turns=turns, r_max=r_max, mean=mean, std=std)
    _, xpred_in = field_and_prediction(model, z1)
    xpred = denormalize(xpred_in, mean, std)
    xp = xpred.detach().cpu()
    gam = gamma.squeeze(-1).detach().cpu()
    sp = spiral_ref.detach().cpu()
    ax.clear()
    ax.plot(sp[::4, 0].numpy(), sp[::4, 1].numpy(), ".", ms=0.5, alpha=0.3, c="gray")
    ax.scatter(xp[:, 0].numpy(), xp[:, 1].numpy(), c=gam.numpy(), s=1.5, alpha=0.5,
               cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title("xpred (blue=low gamma, red=high)")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal", "box")


def plot_energy_landscape(model, device, ax, spiral_ref,
                          lim=3.2, res=120,
                          mean=None, std=None):
    xs = torch.linspace(-lim, lim, res, device=device)
    ys = torch.linspace(-lim, lim, res, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    pts_in = normalize(pts, mean, std)
    fz, _ = field_and_prediction(model, pts_in)
    fz = fz * std
    mag = torch.linalg.norm(fz, dim=-1).reshape(res, res).detach().cpu()

    ax.clear()
    ax.imshow(mag.numpy().T, origin="lower", extent=[-lim, lim, -lim, lim],
              cmap="inferno", aspect="equal",
              norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=mag.max().item()))
    sp = spiral_ref.detach().cpu()
    ax.plot(sp[::6, 0].numpy(), sp[::6, 1].numpy(), ".", ms=0.5, alpha=0.6, c="cyan")
    ax.set_title("||f(z)|| (dark=minima)")


def plot_landing_quality(model, device, ax, spiral_ref,
                         lim=3.2, res=120,
                         mean=None, std=None):
    xs = torch.linspace(-lim, lim, res, device=device)
    ys = torch.linspace(-lim, lim, res, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    pts_in = normalize(pts, mean, std)
    _, xpred_in = field_and_prediction(model, pts_in)
    xpred = denormalize(xpred_in, mean, std)
    d = dist_to_spiral(xpred, spiral_ref).squeeze(-1).reshape(res, res).detach().cpu()

    ax.clear()
    ax.imshow(d.numpy().T, origin="lower", extent=[-lim, lim, -lim, lim],
              cmap="inferno", aspect="equal",
              norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=d.max().item()))
    sp = spiral_ref.detach().cpu()
    ax.plot(sp[::6, 0].numpy(), sp[::6, 1].numpy(), ".", ms=0.5, alpha=0.6, c="cyan")
    ax.set_title("dist(xpred, spiral) dark=good")


def plot_dist_improvement(model, device, ax, spiral_ref, B=4000, turns=3.0,
                          r_max=3.0, mean=None, std=None):
    z1, _, _, gamma = sample_pair_batch(
        B, device, turns=turns, r_max=r_max, mean=mean, std=std)
    _, xpred_in = field_and_prediction(model, z1)
    z1_orig = denormalize(z1, mean, std)
    xpred = denormalize(xpred_in, mean, std)
    dz = dist_to_spiral(z1_orig, spiral_ref).squeeze(-1).detach().cpu()
    dp = dist_to_spiral(xpred, spiral_ref).squeeze(-1).detach().cpu()
    gam = gamma.squeeze(-1).detach().cpu()

    ax.clear()
    ax.scatter(dz.numpy(), dp.numpy(), s=1, alpha=0.3, c=gam.numpy(),
               cmap="coolwarm", vmin=0, vmax=1)
    mx = max(dz.max().item(), dp.max().item()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("dist(z, spiral)")
    ax.set_ylabel("dist(z+f(z), spiral)")
    ax.set_title("below diag = closer after step")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_aspect("equal", "box")


def plot_f_along_spiral(model, device, ax, spiral_ref, mean=None, std=None):
    pts_in = normalize(spiral_ref, mean, std)
    fz, _ = field_and_prediction(model, pts_in)
    fz = fz * std
    mag = torch.linalg.norm(fz, dim=-1).detach().cpu()
    ax.clear()
    ax.plot(mag.numpy(), lw=1.0, c="steelblue")
    ax.set_title("||f|| along spiral")
    ax.set_xlabel("spiral index")
    ax.set_ylabel("||f||")


def save_summary_panel_png(
    model, device, spiral_ref, mean, std,
    out_path, title=None,
    turns=3.0, r_max=3.0,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.subplots_adjust(wspace=0.34, hspace=0.28)
    if title:
        fig.suptitle(title, fontsize=10)

    ax_field, ax_energy, ax_landing, ax_xpred, ax_dist, ax_fspiral = axes.ravel()
    plot_quiver_field(model, device, ax_field, spiral_ref, mean=mean, std=std)
    plot_energy_landscape(model, device, ax_energy, spiral_ref, mean=mean, std=std)
    plot_landing_quality(model, device, ax_landing, spiral_ref, mean=mean, std=std)
    plot_pred_scatter(model, device, ax_xpred, spiral_ref, turns=turns, r_max=r_max, mean=mean, std=std)
    plot_dist_improvement(model, device, ax_dist, spiral_ref, turns=turns, r_max=r_max, mean=mean, std=std)
    plot_f_along_spiral(model, device, ax_fspiral, spiral_ref, mean=mean, std=std)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path



def path_with_metric(path, metric_name, metric_value):
    p = Path(path)
    suffix = p.suffix
    stem = p.stem
    tagged = f"{stem}_{metric_name}{metric_value:.4f}{suffix}"
    return str(p.with_name(tagged))


def extract_tags_from_name(name):
    tau_match = re.search(r"tau([0-9.\-]+)", name)
    alpha_match = re.search(r"alpha([0-9.\-]+)", name)
    tau_tag = f"tau{tau_match.group(1)}" if tau_match else "tauNA"
    alpha_tag = f"alpha{alpha_match.group(1)}" if alpha_match else "alphaNA"
    attr_match = re.search(r"_attr([A-Za-z0-9.\-]+)", name)
    rep_match = re.search(r"_rep([A-Za-z0-9.\-]+)", name)
    if attr_match and rep_match:
        attr_tag = f"attr{attr_match.group(1)}"
        rep_tag = f"rep{rep_match.group(1)}"
    else:
        mode_match = re.search(r"_(inv|none)", name)
        mode = mode_match.group(1) if mode_match else "NA"
        attr_tag = f"attr{mode}"
        rep_tag = f"rep{mode}"
    return tau_tag, alpha_tag, attr_tag, rep_tag


def build_tags(tau, alpha, reweight_attr, reweight_rep):
    return f"tau{tau:.3f}_alpha{alpha:.3f}_attr{reweight_attr}_rep{reweight_rep}"


def build_eval_video_name(ckpt_path, step_size, num_steps, out_dir=None):
    base = Path(ckpt_path).name
    tau_tag, alpha_tag, attr_tag, rep_tag = extract_tags_from_name(base)
    eta_tag = f"eta{step_size:.3f}"
    steps_tag = f"steps{num_steps}"
    fname = f"rollout_{tau_tag}_{alpha_tag}_{attr_tag}_{rep_tag}_{eta_tag}_{steps_tag}.mp4"
    if out_dir is None:
        return str(Path(ckpt_path).with_name(fname))
    return str(Path(out_dir) / fname)


def build_summary_png_name(ckpt_path, out_dir=None):
    base = Path(ckpt_path).name
    tau_tag, alpha_tag, attr_tag, rep_tag = extract_tags_from_name(base)
    fname = f"summary_{tau_tag}_{alpha_tag}_{attr_tag}_{rep_tag}.png"
    if out_dir is None:
        return str(Path(ckpt_path).with_name(fname))
    return str(Path(out_dir) / fname)


@torch.no_grad()
def save_noise_rollout_video(
    model, device, spiral_ref,
    mean, std,
    n_points=100000, n_steps=5, step_size=0.2,
    out_path="noise_rollout.mp4", fps=6,
    metric_chunk=2048,
    metric_name="cd",
    tag_output_with_metric=True,
    lim=3.2,
):
    model.eval()

    z = torch.randn(n_points, 2, device=device)
    states = [denormalize(z, mean, std).detach().cpu()]
    for _ in range(n_steps):
        f, _ = field_and_prediction(model, z)
        z = z + step_size * f
        states.append(denormalize(z, mean, std).detach().cpu())

    z_final = denormalize(z, mean, std)
    eval_metric = symmetric_chamfer_distance(z_final, spiral_ref, chunk=metric_chunk).item()
    target_path = path_with_metric(out_path, metric_name, eval_metric) if tag_output_with_metric else out_path

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111)
    sp = spiral_ref.detach().cpu()
    ax.plot(sp[::4, 0].numpy(), sp[::4, 1].numpy(), ".", ms=0.6, alpha=0.35, c="gray")
    scat = ax.scatter(states[0][:, 0].numpy(), states[0][:, 1].numpy(),
                      s=2, alpha=0.45, c="steelblue")
    title = ax.set_title(f"rollout step 0/{n_steps}  eta={step_size}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")

    def update(frame):
        pts = states[frame].numpy()
        scat.set_offsets(pts)
        title.set_text(f"rollout step {frame}/{n_steps}  eta={step_size}")
        return scat, title

    ani = animation.FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)

    saved_path = None
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(target_path, writer=writer)
        saved_path = target_path
    except Exception:
        gif_path = str(Path(target_path).with_suffix(".gif"))
        try:
            writer = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writer)
            saved_path = gif_path
        except Exception:
            saved_path = None

    plt.close(fig)
    return saved_path, eval_metric
