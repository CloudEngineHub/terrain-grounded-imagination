"""Command-line options of ``scripts/train.py``. The defaults are the full recipe.

This module does not import torch or Isaac Lab, so the options are parsed before the simulator starts.
"""

from __future__ import annotations

import argparse
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the project root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Terrain Grounded Imagination: train an ANYmal-D locomotion policy through a world model.")

    g = p.add_argument_group("run")
    g.add_argument("--log_dir", type=str, default=None, help="output directory (default: logs/<date_time>)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--iterations", type=int, default=30_000)
    g.add_argument("--resume", type=str, default=None, help="checkpoint (last.pt) to continue from")
    g.add_argument("--max_hours", type=float, default=1e9, help="stop and save after this long (this job's time only)")
    g.add_argument("--ckpt_every", type=int, default=2000)
    g.add_argument("--log_every", type=int, default=50)

    g = p.add_argument_group("simulator and curriculum")
    g.add_argument("--num_envs", type=int, default=4096)
    g.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
    g.add_argument("--buffer_steps", type=int, default=200, help="replay capacity: buffer_steps x num_envs transitions")
    g.add_argument("--warmup_steps", type=int, default=64, help="simulator steps collected before the first update")
    g.add_argument("--action_noise", type=float, default=0.1, help="exploration noise of the simulator actions")
    g.add_argument("--stepping_stones", type=float, default=0.3, help="share of stepping-stone terrain")
    g.add_argument("--pit_step", type=float, default=0.05,
                   help="weight of the step-out-of-a-pit terrain (2 x 2 m, 5 to 40 cm deep) among the terrains; 0 = off")
    g.add_argument("--parkour", type=str, default="climb_up:6,climb_down:6,gap:6",
                   help="extra parkour columns 'family:cols,...': climb_up (pit, wall 0.1-1.0 m), climb_down (box, "
                        "0.1-1.0 m), gap (0.1-1.0 m), as in ANYmal Parkour's climb / jump skills")
    g.add_argument("--sgs_spawns", type=int, default=4, help="curriculum: spawn patches per terrain cell (squared)")
    g.add_argument("--sgs_dirs", type=int, default=4, help="curriculum: travel-direction bins per terrain cell")
    g.add_argument("--sgs_neighbor", type=float, default=0.25,
                   help="curriculum: a configuration's success counts as at least this times that of the same "
                        "configuration one level easier")
    g.add_argument("--sgs_target", type=float, default=0.8, help="curriculum: success rate the terrain sampler aims for")
    g.add_argument("--sgs_give", type=float, default=0.5,
                   help="curriculum: a solved terrain type gives away up to this fraction of its share to types that "
                        "still need it (0 = fixed type shares)")
    g.add_argument("--sgs_window", type=int, default=16,
                   help="curriculum: outcomes kept per configuration (a configuration is visited about once per 1k iterations)")

    g = p.add_argument_group("world model")
    g.add_argument("--wm_hidden", type=int, nargs="+", default=[1024, 1024, 1024])
    g.add_argument("--wm_window", type=int, default=8, help="autoregressive training window (steps)")
    g.add_argument("--wm_batches", type=int, default=8, help="world-model updates per iteration")
    g.add_argument("--wm_batch", type=int, default=2048)
    g.add_argument("--contact_w", type=float, default=0.5, help="weight of the contact (loaded / not loaded) loss")
    g.add_argument("--force_w", type=float, default=0.5,
                   help="weight of the contact-force loss (squared error, in units of 100 N)")
    g.add_argument("--lr", type=float, default=2e-4, help="learning rate of the world model and the critic")
    g.add_argument("--muon_min_dim", type=int, default=1,
                   help="world model: weight matrices with both sides >= this go to Muon (1: all matrices)")
    g.add_argument("--muon_lr_mult", type=float, default=1.0, help="Muon learning rate = mult x Adam's (RMS-matched)")

    g = p.add_argument_group("policy learning in imagination")
    g.add_argument("--horizon", type=int, default=32, help="imagined steps (20 ms each)")
    g.add_argument("--pi_batch", type=int, default=2048, help="imagined rollouts per policy update")
    g.add_argument("--pi_lr", type=float, default=2e-4)
    g.add_argument("--pi_muon_mult", type=float, default=4.0,
                   help="policy Muon step multiple at the start, falling linearly to x1 over --pi_mult_decay")
    g.add_argument("--pi_mult_decay", type=int, default=1700,
                   help="iterations (from the start / resume) over which the policy Muon multiple falls linearly to x1; "
                        "0: constant")
    g.add_argument("--pi_ema", type=float, default=0.995,
                   help="deployed policy = exponential moving average of the trained one with this decay; saved as "
                        "pi_ema, used by evaluation")
    g.add_argument("--lr_decay_iters", type=int, default=100_000, help="cosine decay to 0.1 x the learning rates")
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--lam", type=float, default=0.95, help="lambda of the critic targets")
    g.add_argument("--pg_event", type=float, default=0.3,
                   help="cut the gradient through the model at imagined touchdowns whose unsupported footprint share exceeds "
                        "this (bootstrap from the critic there)")
    g.add_argument("--lr_coef", type=float, default=0.1, help="weight of the likelihood-ratio (advantage) term")
    g.add_argument("--imag_noise", type=float, default=0.2, help="action noise in imagination when --lr_coef > 0")
    g.add_argument("--sym_coef", type=float, default=2.0, help="weight of the left/right mirror-consistency loss")
    g.add_argument("--critic_epochs", type=int, default=5)
    g.add_argument("--critic_minibatches", type=int, default=4)
    g.add_argument("--wm_only_iters", type=int, default=300, help="iterations that train only the world model")

    g = p.add_argument_group("perception")
    g.add_argument("--fpm_bootstrap", type=int, default=1500, help="iterations with fixed footholds before the FPM takes over")
    g.add_argument("--fpm_label_every", type=int, default=25, help="iterations between foothold-label refreshes")
    g.add_argument("--fpm_steps", type=int, default=8, help="FPM updates per iteration")
    g.add_argument("--fpm_batch", type=int, default=4096)

    g = p.add_argument_group("rewards and terminations (AME2 goal-reaching reward, tgi_anymal/mdp/mdp.py RewardCfg)")
    g.add_argument("--episode_s", type=float, default=8.0, help="episode length = time budget to reach the goal (s)")
    g.add_argument("--goal_dist", type=float, nargs=2, default=(2.5, 3.5), help="goal distance from the spawn cell (m)")
    g.add_argument("--goal_tol", type=float, default=0.5, help="curriculum success: final distance to the goal (m)")
    g.add_argument("--w_self", type=float, default=-2.0,
                   help="leg-leg self-collision proximity (spheres / feet of different legs closer than --self_margin)")
    g.add_argument("--self_margin", type=float, default=0.05)
    g.add_argument("--w_face", type=float, default=2.0, help="face the goal while > 0.5 m from it (forward locomotion)")
    g.add_argument("--w_lat", type=float, default=-1.0, help="lateral base velocity squared (forward locomotion)")
    g.add_argument("--w_impact", type=float, default=-3.0,
                   help="foot impact penalty: sum_feet max(|F| - impact_n, 0) / 100 N, F from the world model")
    g.add_argument("--impact_n", type=float, default=600.0, help="foot force (N) above which --w_impact applies")
    g.add_argument("--w_power", type=float, default=-0.002, help="mechanical power penalty per W: sum_j |tau_j qd_j| (PD-law tau)")
    g.add_argument("--w_term", type=float, default=-500.0, help="termination penalty per second (x dt)")
    g.add_argument("--term_floor", type=float, default=-2.5,
                   help="terminate when the base is below this absolute height (m): fell into a floorless gap")
    g.add_argument("--term_angle", type=float, default=1.4, help="tilt termination (rad)")
    g.add_argument("--term_height", type=float, default=0.25, help="terminate when the base is below this above the terrain (m)")
    g.add_argument("--contact_roles", type=str, default="knee:tolerated,shank:tolerated,thigh:allowed",
                   help="'part:role,...' over the collision spheres' names (assets/anymal_d_kinematics_col.xml); allowed "
                        "(no penalty), tolerated (contact penalty) or forbidden (ends the episode, the default: the belly)")
    g.add_argument("--term_clear", type=float, default=0.0,
                   help="terminate when a forbidden sphere is closer than this to the terrain (m)")

    g = p.add_argument_group("heightmap")
    g.add_argument("--hm_res", type=float, default=0.04, help="heightmap resolution (m)")
    g.add_argument("--field_window", type=float, default=0.5, help="support field: neighbourhood for the local top (m)")
    g.add_argument("--field_hole", type=float, default=0.25, help="support field: deeper than this below the local top is a gap (m)")

    g = p.add_argument_group("speed")
    g.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"],
                   help="mixed precision; GPUs without native bf16 fall back to fp32")
    g.add_argument("--compile", type=int, default=1, help="torch.compile the update functions")
    g.add_argument("--cudagraphs", type=int, default=1, help="replay the world-model update as CUDA graphs")
    g.add_argument("--cg_imag", type=int, default=1, help="replay the imagination step as CUDA graphs")
    g.add_argument("--async_collect", type=int, default=1,
                   help="run the simulator in a thread, overlapping the updates (it acts with networks one iteration old)")
    g.add_argument("--async_after", type=int, default=1600, help="iteration from which the simulator overlaps the updates")
    g.add_argument("--vram_gb", type=float, default=30.0, help="GPU memory budget of the whole run (simulator + torch)")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = build_parser().parse_args(argv)
    if cfg.log_dir is None:
        cfg.log_dir = os.path.join(REPO, "logs", time.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg.log_dir = os.path.abspath(cfg.log_dir)
    return cfg
