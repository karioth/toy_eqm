import argparse
import os

import torch

from utils import (
    build_eval_video_name,
    build_summary_png_name,
    build_model,
    enable_tf32_if_cuda,
    extract_tags_from_name,
    get_device,
    make_spiral_reference,
    save_noise_rollout_video,
    save_summary_panel_png,
    spiral_normalization,
)


def _load_model(ckpt_path, device):
    model = build_model(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def render_summary_png_from_checkpoint(
    ckpt_path,
    turns=3.0,
    r_max=3.0,
    n_ref=12000,
    out_dir=None,
):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    enable_tf32_if_cuda(device)
    model = _load_model(ckpt_path, device)

    spiral_ref = make_spiral_reference(device, n_ref=n_ref, turns=turns, r_max=r_max)
    mean, std = spiral_normalization(spiral_ref)

    tau_tag, alpha_tag, attr_tag, rep_tag = extract_tags_from_name(os.path.basename(ckpt_path))
    title = f"{tau_tag}  {alpha_tag}  {attr_tag}  {rep_tag}"
    out_path = build_summary_png_name(ckpt_path, out_dir=out_dir)
    return save_summary_panel_png(
        model,
        device,
        spiral_ref,
        mean,
        std,
        out_path,
        title=title,
        turns=turns,
        r_max=r_max,
    )


def render_eval_video_from_checkpoint(
    ckpt_path,
    step_size=0.2,
    num_samples=100000,
    num_steps=5,
    fps=6,
    turns=3.0,
    r_max=3.0,
    n_ref=12000,
    out_dir=None,
):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    enable_tf32_if_cuda(device)
    model = _load_model(ckpt_path, device)

    spiral_ref = make_spiral_reference(device, n_ref=n_ref, turns=turns, r_max=r_max)
    mean, std = spiral_normalization(spiral_ref)

    out_path = build_eval_video_name(ckpt_path, step_size, num_steps, out_dir=out_dir)
    return save_noise_rollout_video(
        model,
        device,
        spiral_ref,
        mean,
        std,
        n_points=num_samples,
        n_steps=num_steps,
        step_size=step_size,
        out_path=out_path,
        fps=fps,
    )


def _parse_args():
    p = argparse.ArgumentParser(description="Render rollout video + summary PNG from a checkpoint.")
    p.add_argument("ckpt", type=str, help="Path to checkpoint (.pt)")
    p.add_argument("--step-size", type=float, default=0.2)
    p.add_argument("--num-samples", type=int, default=100000)
    p.add_argument("--num-steps", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    summary_path = render_summary_png_from_checkpoint(ckpt_path)
    print(f"saved summary: {summary_path}")

    eval_path, eval_metric = render_eval_video_from_checkpoint(
        ckpt_path,
        step_size=args.step_size,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
    )
    print(f"eval cd: {eval_metric:.6f}")
    if eval_path is None:
        print("warning: could not save rollout animation")
    else:
        print(f"saved rollout animation: {eval_path}")
