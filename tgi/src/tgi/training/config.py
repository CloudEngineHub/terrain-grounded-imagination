"""Command-line options of ``scripts/train.py``. The defaults are the full recipe.

This module does not import torch or Isaac Lab, so the options are parsed before the simulator starts.
"""

from __future__ import annotations

import argparse
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the tgi project


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Terrain Grounded Imagination: train a G1 locomotion policy through a world model.")

    g = p.add_argument_group("run")
    g.add_argument("--log_dir", type=str, default=None, help="output directory (default: logs/<date_time>)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--iterations", type=int, default=40_000)
    g.add_argument("--resume", type=str, default=None, help="checkpoint (last.pt) to continue from")
    g.add_argument("--max_hours", type=float, default=1e9, help="stop and save after this long (this job's time only)")
    g.add_argument("--ckpt_every", type=int, default=2000)
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--eval_every", type=int, default=0,
                   help="full background check (all courses + base env validation, CPU, about 40 min); 0 = off")
    g.add_argument("--quick_every", type=int, default=0,
                   help="quick background course check (about 30 s, CPU), a multiple of --ckpt_every; 0 = off")

    g = p.add_argument_group("simulator and curriculum")
    g.add_argument("--num_envs", type=int, default=8192)
    g.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
    g.add_argument("--buffer_steps", type=int, default=200, help="replay capacity: buffer_steps x num_envs transitions")
    g.add_argument("--warmup_steps", type=int, default=64, help="simulator steps collected before the first update")
    g.add_argument("--action_noise", type=float, default=0.1, help="exploration noise of the simulator actions")
    g.add_argument("--stepping_stones", type=float, default=0.6, help="share of stepping-stone terrain")
    g.add_argument("--pit_step", type=float, default=0.05,
                   help="weight of the step-out-of-a-pit terrain (2 x 2 m, 5 to 40 cm deep) among the terrains; 0 = off")
    g.add_argument("--sgs_spawns", type=int, default=4, help="curriculum: spawn patches per terrain cell (squared)")
    g.add_argument("--sgs_dirs", type=int, default=4, help="curriculum: travel-direction bins per terrain cell")
    g.add_argument("--sgs_neighbor", type=float, default=0.25,
                   help="curriculum: a configuration's success counts as at least this times that of the same "
                        "configuration one level easier")
    g.add_argument("--sgs_target", type=float, default=0.8, help="curriculum: success rate the terrain sampler aims for")
    g.add_argument("--sgs_window", type=int, default=16,
                   help="curriculum: outcomes kept per configuration (a configuration is visited about once per 1k iterations)")

    g = p.add_argument_group("world model")
    g.add_argument("--wm_hidden", type=int, nargs="+", default=[1024, 1024, 1024])
    g.add_argument("--wm_window", type=int, default=8, help="autoregressive training window (steps)")
    g.add_argument("--wm_batches", type=int, default=8, help="world-model updates per iteration")
    g.add_argument("--wm_batch", type=int, default=2048)
    g.add_argument("--contact_w", type=float, default=0.5, help="weight of the foot-contact loss")
    g.add_argument("--force_w", type=float, default=0.5,
                   help="weight of the sole contact-force loss (Huber, in units of 100 N)")
    g.add_argument("--lr", type=float, default=2e-4, help="learning rate of the world model and the critic")
    g.add_argument("--muon_min_dim", type=int, default=1,
                   help="world model: weight matrices with both sides >= this go to Muon (1: all matrices)")
    g.add_argument("--muon_lr_mult", type=float, default=1.0,
                   help="world model: Muon learning rate = mult x Adam's (RMS-matched)")

    g = p.add_argument_group("policy learning in imagination")
    g.add_argument("--horizon", type=int, default=32, help="imagined steps (20 ms each)")
    g.add_argument("--pi_batch", type=int, default=2048, help="imagined rollouts per policy update")
    g.add_argument("--pi_lr", type=float, default=2e-4)
    g.add_argument("--pi_muon_mult", type=float, default=4.0,
                   help="policy: Muon learning rate = mult x Adam's at the start, falling linearly to x1 over "
                        "--pi_mult_decay iterations")
    g.add_argument("--pi_mult_decay", type=int, default=1700,
                   help="iterations over which the policy's Muon multiple falls linearly to x1; 0: constant")
    g.add_argument("--pi_ema", type=float, default=0.995,
                   help="the deployed policy is an exponential moving average of the trained one with this decay "
                        "(saved as pi_ema, used by the export)")
    g.add_argument("--lr_decay_iters", type=int, default=100_000,
                   help="cosine decay to 0.1 x the learning rates over this many iterations, whatever --iterations is")
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--lam", type=float, default=0.95, help="lambda of the critic targets")
    g.add_argument("--pg_event", type=float, default=0.3,
                   help="cut the gradient through the model at imagined touchdowns whose unsupported sole share exceeds "
                        "this (bootstrap from the critic there); 0 = plain H-step return + critic bootstrap")
    g.add_argument("--lr_coef", type=float, default=0.1, help="weight of the likelihood-ratio (advantage) term")
    g.add_argument("--imag_noise", type=float, default=0.2, help="action noise in imagination when --lr_coef > 0")
    g.add_argument("--imag_stand", type=float, default=0.1, help="share of imagined rollouts with a zero command")
    g.add_argument("--sym_coef", type=float, default=2.0, help="weight of the left/right mirror-consistency loss")
    g.add_argument("--critic_epochs", type=int, default=5)
    g.add_argument("--critic_minibatches", type=int, default=4)
    g.add_argument("--wm_only_iters", type=int, default=300, help="iterations that train only the world model")

    g = p.add_argument_group("foothold prediction model")
    g.add_argument("--fpm_bootstrap", type=int, default=1500, help="iterations with fixed footholds before the FPM takes over")
    g.add_argument("--fpm_label_every", type=int, default=25, help="iterations between foothold-label refreshes")
    g.add_argument("--fpm_steps", type=int, default=8, help="FPM updates per iteration")
    g.add_argument("--fpm_batch", type=int, default=4096)

    g = p.add_argument_group("rewards and terminations")
    g.add_argument("--w_gait", type=float, default=0.25)
    g.add_argument("--w_foothold", type=float, default=-3.0, help="contact x (unsupported sole share + 1 - support field)")
    g.add_argument("--w_upright", type=float, default=0.25)
    g.add_argument("--w_pose", type=float, default=2.5, help="standing commands only")
    g.add_argument("--w_impact", type=float, default=-1.0,
                   help="foot impact penalty: sum_soles max(|F| - impact_n, 0) / 100 N, F from the world model")
    g.add_argument("--impact_n", type=float, default=450.0, help="sole force (N) above which --w_impact applies")
    g.add_argument("--lava_clear", type=float, default=0.05,
                   help="foot clearance (m) above the lava floor of the stepping-stone pits that terminates")
    g.add_argument("--w_dev_arm_side", type=float, default=-1.0, help="extra |deviation| penalty on shoulder roll / yaw")
    g.add_argument("--w_still", type=float, default=-1.0, help="standing commands: per foot out of contact")
    g.add_argument("--w_dev_waist", type=float, default=-1.0)
    g.add_argument("--w_edge", type=float, default=-0.5, help="stance sole share on height edges (nosings, rims)")
    g.add_argument("--w_flat", type=float, default=-300.0, help="stance sole-to-terrain gap variance (m^2): flat landing")
    g.add_argument("--w_overspeed", type=float, default=-4.0, help="(|v_xy| - |cmd_xy| - 0.1)_+^2")
    g.add_argument("--w_torso_roll", type=float, default=-20.0, help="torso roll sin^2 penalty")
    g.add_argument("--w_torso_roll_rate", type=float, default=-1.0, help="torso roll rate^2 penalty (pelvis + waist roll)")
    g.add_argument("--w_dev_hip", type=float, default=-1.0)
    g.add_argument("--w_dev_arms", type=float, default=-0.1)
    g.add_argument("--term_hole", type=float, default=0.3,
                   help="terminate when the pelvis is less than this above the highest nearby terrain (m)")
    g.add_argument("--term_angle", type=float, default=0.7, help="imagined tilt termination (rad), the base env's limit")

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
    g.add_argument("--phys", type=str, default="contacts_per_env=100,nconmax=100,njmax=500",
                   help="Newton solver / collision budgets (key=value,...; contacts_per_env -> rigid_contact_max); the "
                        "auto-sized defaults cost ~1.45 GB at 4096 robots. '' keeps Isaac Lab's preset")
    g.add_argument("--vram_gb", type=float, default=22.0, help="GPU memory budget of the whole run (simulator + torch)")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = build_parser().parse_args(argv)
    if cfg.log_dir is None:
        cfg.log_dir = os.path.join(REPO, "logs", time.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg.log_dir = os.path.abspath(cfg.log_dir)
    return cfg
