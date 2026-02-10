import os
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from utils import (
    attraction_weight,
    build_tags,
    build_model,
    enable_tf32_if_cuda,
    field_and_prediction,
    get_device,
    make_spiral_reference,
    sample_pair_batch,
    spiral_normalization,
)
from eval_toy import render_eval_video_from_checkpoint, render_summary_png_from_checkpoint


def huber_mean(e, delta):
    if delta <= 0:
        raise ValueError("huber_delta must be > 0")
    abs_e = e.abs()
    quadratic = 0.5 * e * e
    linear = delta * (abs_e - 0.5 * delta)
    return torch.where(abs_e <= delta, quadratic, linear).mean(dim=-1)


def train_live(
    steps=25000,
    B=8192,
    lr=1e-2,
    turns=3.0,
    r_max=3.0,
    n_ref=12000,
    alpha=0.0,
    tau=1.0,
    huber_delta=1.0,
    reweight_attr="none",
    reweight_rep="none",
    seed=0,
    show_progress=True,
    out_dir=".",
    eval_step_size=0.2,
    eval_num_samples=100000,
    eval_num_steps=5,
):
    if reweight_attr not in ("inv", "none"):
        raise ValueError("reweight_attr must be 'inv' or 'none'")
    if reweight_rep not in ("inv", "none"):
        raise ValueError("reweight_rep must be 'inv' or 'none'")

    torch.manual_seed(seed)
    device = get_device()
    enable_tf32_if_cuda(device)
    print(f"device: {device}")
    print(
        "alpha={alpha}, tau={tau}, huber_delta={huber_delta}, "
        "reweight_attr={reweight_attr}, reweight_rep={reweight_rep}".format(
            alpha=alpha,
            tau=tau,
            huber_delta=huber_delta,
            reweight_attr=reweight_attr,
            reweight_rep=reweight_rep,
        )
    )

    # alpha only affects attraction/repulsion weights in this Huber variant.
    tag = f"huber{huber_delta:.3f}_" + build_tags(tau, alpha, reweight_attr, reweight_rep)
    run_dir = Path(out_dir) / tag
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = str(run_dir / f"ckpt_{tag}.pt")

    spiral_ref = make_spiral_reference(device, n_ref=n_ref, turns=turns, r_max=r_max)
    mean, std = spiral_normalization(spiral_ref, eps=1e-6)
    print(f"norm mean={mean.squeeze(0).tolist()} std={std.squeeze(0).tolist()}")

    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pbar = None
    step_iter = range(1, steps + 1)
    if show_progress:
        if tqdm is None:
            print("warning: tqdm not available; progress bar disabled")
        else:
            pbar = tqdm(step_iter, total=steps, desc="train", dynamic_ncols=True)
            step_iter = pbar

    ema = None
    for step in step_iter:
        z1, z2, x, gamma = sample_pair_batch(
            B, device, turns=turns, r_max=r_max, mean=mean, std=std
        )
        f1, _ = field_and_prediction(model, z1)
        f2, _ = field_and_prediction(model, z2)
        target1 = x - z1
        target2 = x - z2
        e1 = f1 - target1
        e2 = f2 - target2
        w_attr = attraction_weight(gamma, mode=reweight_attr).pow(alpha)
        w_rep = attraction_weight(gamma, mode=reweight_rep).pow(alpha)

        d1 = huber_mean(e1, huber_delta)
        d2 = huber_mean(e2, huber_delta)
        attract = (w_attr * (d1 + d2)).mean()
        if tau != 0.0:
            d12 = huber_mean(e1 - e2, huber_delta)
            rep = (w_rep * d12).mean()
            loss = attract - tau * rep
        else:
            rep = torch.tensor(0.0, device=device)
            loss = attract

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        l = loss.detach()
        ema = l if ema is None else 0.99 * ema + 0.01 * l
        if pbar is not None and (step % 20 == 0 or step == steps):
            pbar.set_postfix(
                loss=f"{l.item():.4f}",
                ema=f"{ema.item():.4f}",
                attract=f"{attract.item():.3f}",
                rep=f"{rep.item():.3f}",
            )

    if pbar is not None:
        pbar.close()

    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")

    summary_path = render_summary_png_from_checkpoint(
        ckpt_path,
        turns=turns,
        r_max=r_max,
        n_ref=n_ref,
        out_dir=str(run_dir),
    )
    print(f"saved summary: {summary_path}")

    eval_path, eval_metric = render_eval_video_from_checkpoint(
        ckpt_path,
        step_size=eval_step_size,
        num_samples=eval_num_samples,
        num_steps=eval_num_steps,
        turns=turns,
        r_max=r_max,
        n_ref=n_ref,
        out_dir=str(run_dir),
    )
    print(f"eval cd: {eval_metric:.6f}")
    if eval_path is None:
        print("warning: could not save rollout animation")
    else:
        print(f"saved rollout animation: {eval_path}")

    return model, ckpt_path


if __name__ == "__main__":
    train_live(
        tau=1.0,
        reweight_attr="inv",
        reweight_rep="none",
        eval_step_size=0.7,
        eval_num_samples=100000,
        eval_num_steps=20,
        alpha=1,
        huber_delta=1.0,
        seed=123,
    )
