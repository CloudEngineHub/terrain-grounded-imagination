"""Differentiable MDP heads shared by the simulator and imagination: state, rewards and terminations.

State ``S = (p, R, v, w, q, qd)``: pelvis position, rotation, linear and angular velocity, joint positions and
velocities. Contact-dependent rewards use the world model's foot-contact probabilities and the terrain under the feet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .interface import JointMap, policy_proprio, system_state
from .kinematics import G1Kinematics
from .rotations import mat_to_6d, quat_to_mat, yaw_mat
from ..constants import NUM_J, POSE_STD, match_values
from .terrain import TerrainFeatures

INV_DIM = 6 + 3 + 3 + NUM_J + NUM_J  # height comes from the terrain features


@dataclass
class RewardCfg:
    # foot impact, on the world model's sole forces K["ff"]: sum over soles of max(|F| - impact_n, 0) / 100 N
    w_impact: float = 0.0
    impact_n: float = 450.0
    w_lin: float = 2.0
    std_lin: float = math.sqrt(0.25)
    w_ang: float = 2.0
    std_ang: float = math.sqrt(0.5)
    w_upright: float = 0.25
    std_upright: float = math.sqrt(0.2)
    w_pose: float = 2.5  # standing commands only (the base env's stand_still)
    w_limits: float = -1.0
    w_action_rate: float = -0.1
    w_clearance: float = -2.0  # only swing feet whose lowest sole point is below the target height above the terrain
    clearance_target: float = 0.1
    w_slip: float = -0.1
    w_body_ang_vel: float = -0.05
    w_gait: float = 0.25
    w_foothold: float = -3.0  # contact x (unsupported sole share + 1 - support field)
    w_still: float = -1.0  # standing commands: per foot out of contact
    # the base env's joint-deviation L1 regularizers (fpm/env_cfg.py RewardsCfg)
    w_dev_waist: float = -1.0
    w_dev_hip: float = -1.0
    w_dev_arms: float = -0.1
    w_dev_arm_side: float = 0.0  # extra |deviation| weight on shoulder roll / yaw (arms swing fore-aft, off the body)
    # torso roll sway: sin^2 of the torso roll (lateral gravity in the torso frame) and the torso roll rate
    # (pelvis body roll rate + waist roll joint velocity), rad^2 / (rad/s)^2
    # stairs / footholds: sole share on edges in stance, sole not parallel to the terrain under it in stance
    # (variance of the sole-to-terrain gap over the 15 sole points, m^2), and speed above the command
    w_edge: float = 0.0
    w_flat: float = 0.0
    w_overspeed: float = 0.0
    w_torso_roll: float = 0.0
    w_torso_roll_rate: float = 0.0
    lava_clear: float = 0.05  # >0: terminate when a foot over a lava floor (stones-tile pit) is within this of it
    # tracking on v_xy and w_z, with weak lin_vel_z and ang_vel_xy penalties as in the base env
    w_lin_z: float = -0.5
    w_ang_xy: float = -0.05
    cmd_threshold: float = 0.05
    walking_threshold: float = 0.05
    running_threshold: float = 1.5
    term_angle: float = math.radians(70.0)
    term_height: float = 0.3
    term_hole: float = 0.3  # >0: terminate when the pelvis is less than this above the highest nearby terrain


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


def invariant_features(S) -> torch.Tensor:
    """Heading/position-invariant features: tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (mat_to_6d(H.transpose(-1, -2) @ R), (Rt @ v.unsqueeze(-1)).squeeze(-1), (Rt @ w.unsqueeze(-1)).squeeze(-1),
         q, qd), -1)  # fmt: skip


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Dynamics-relevant context only; the encoder bias (xi[4:33]) only affects observations."""
    e = xi[..., 4 + NUM_J :]
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:4] / 0.03, e[..., 0:1] / 0.15 - 1, e[..., 1:2] / 0.075 - 1,
                      (e[..., 2:3] - 1.1) / 0.4, e[..., 3:4] / 2 - 1, (e[..., 4:5] - 1) / 2), -1)  # fmt: skip


XI_N_DIM = 9


class MDP:
    def __init__(self, kin: G1Kinematics, terrain: TerrainFeatures, default_q: torch.Tensor, joint_names: list[str],
                 soft_lo: torch.Tensor, soft_hi: torch.Tensor, action_scale: torch.Tensor,
                 cfg: RewardCfg = RewardCfg()):
        self.kin, self.terrain, self.cfg = kin, terrain, cfg
        self.default_q = default_q
        self.action_scale = action_scale
        self.soft_lo, self.soft_hi = soft_lo, soft_hi
        dev = default_q.device
        self.jmap = JointMap(joint_names, dev)
        import re

        def mask(pat):
            return torch.tensor([1.0 if re.fullmatch(pat, n) else 0.0 for n in joint_names], device=dev)

        self.m_waist = mask(r"waist.*")
        self.m_hip = mask(r".*_hip_(roll|yaw)_joint")
        self.m_arms = mask(r".*_(shoulder_.*|elbow|wrist_.*)_joint")
        self.m_arm_side = mask(r".*_shoulder_(roll|yaw)_joint")
        self.m_waist_roll = mask(r"waist_roll_joint")
        print(f"[mdp] waist roll joints: {int(self.m_waist_roll.sum())}", flush=True)
        self.q0_isaac = self.jmap.to_isaac(default_q)
        self.std = {k: torch.tensor(match_values(POSE_STD[k], joint_names), device=dev)
                    for k in ("std_standing", "std_walking", "std_running")}
        self.critic_dim = INV_DIM + terrain.wm_dim + 2 + 6 + 3 + XI_N_DIM + terrain.critic_dim

    def _fk(self, S):
        p, R, v, w, q, qd = S
        return self.kin.forward(p, R, v, w, q, qd)

    def context(self, S) -> dict:
        p, R = S[0], S[1]
        fp, fv, rot, bpos = self._fk(S)
        foot_R = rot[..., 1:3, :, :]
        tf, clear, rough, unsup = self.terrain.wm_features(p, R, fp, foot_R)
        K = {"fp": fp, "fv": fv, "rot": rot, "tf": tf, "clear": clear, "rough": rough, "unsup": unsup, "hb": tf[..., 0],
             "torso_p": bpos[..., 0, :], "torso_R": rot[..., 0, :, :], "hole": self.terrain.hole_margin(p, R)}
        K["geo"] = self._geo(S, fp, foot_R)
        # the sole's 5 x 3 points: lowest point's gap to the terrain (swing clearance), gap variance (flat landing)
        # and edge share
        T = self.terrain
        pts = T._points(fp[..., :2], T._yaw2(foot_R), T.foot_grid)  # (..., 2, 15, 2)
        h = T.hm.query(pts)
        g = T.foot_grid  # (15, 2) sole points in the foot frame
        sole_z = fp[..., 2:3] + foot_R[..., 2, 0:1] * g[:, 0] + foot_R[..., 2, 1:2] * g[:, 1]
        gap = sole_z - h
        K["gap_min"] = gap.gather(-1, gap.detach().argmin(-1, keepdim=True)).squeeze(-1)  # argmin + gather (compile-safe)
        K["flat"] = ((gap - gap.mean(-1, keepdim=True)).clamp(-0.05, 0.05) ** 2).mean(-1)  # overhangs saturate
        K["edge"] = T.hm.query_edge(pts).mean(-1)
        return K

    def wm_inputs(self, S):
        """World-model inputs from the state: terrain features and the feet's contact geometry."""
        p, R = S[0], S[1]
        fp, fv, rot, bpos = self._fk(S)
        tf = self.terrain.wm_features(p, R, fp, rot[..., 1:3, :, :])[0]
        return tf, self._geo(S, fp, rot[..., 1:3, :, :])

    def _geo(self, S, fp, foot_R) -> dict:
        """Sole contact geometry + the hinge frames (columns of J), for J^T of the contact wrenches."""
        ax, org = self.kin.joint_frames(S[0], S[1], S[4])
        return dict(self.terrain.foot_contact_geometry(fp, foot_R), ax=ax, org=org)

    def wm_terrain(self, S) -> torch.Tensor:
        """Only the world model's terrain input ``context(S)["tf"]``: skips the eager-only hole margin and the rest."""
        p, R = S[0], S[1]
        fp, fv, rot, bpos = self._fk(S)
        return self.terrain.wm_features(p, R, fp, rot[..., 1:3, :, :])[0]

    def proprio(self, S, a_prev_mj, cmd, xi, noise: bool = True) -> torch.Tensor:
        """Policy proprioception (99): sim encoder bias xi[4:33] corrupts the joint positions."""
        return policy_proprio(S, self.jmap.to_isaac(a_prev_mj), cmd, self.q0_isaac, self.jmap,
                              bias_mj=xi[..., 4 : 4 + NUM_J], noise=noise)

    def system_state(self, S, cmd) -> torch.Tensor:
        return system_state(S, cmd, self.q0_isaac, self.jmap)

    def joint_target(self, a_mj: torch.Tensor) -> torch.Tensor:
        """Joint position setpoint, the action-convention-free world-model input."""
        return self.default_q + self.action_scale * a_mj

    def reward(self, S, K, c, a, a_prev, cmd, dt: float):
        """Per-step reward (scaled by dt). ``c``: foot contact probabilities (..., 2)."""
        cf = self.cfg
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
        w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
        fv, rot = K["fv"], K["rot"]
        speed = cmd[..., :2].norm(dim=-1) + cmd[..., 2].abs()
        active = (speed > cf.cmd_threshold).float()

        e_lin = ((cmd[..., :2] - v_b[..., :2]) ** 2).sum(-1)
        r_lin = torch.exp(-e_lin / cf.std_lin**2)
        e_ang = (cmd[..., 2] - w_b[..., 2]) ** 2
        r_ang = torch.exp(-e_ang / cf.std_ang**2)
        r_extra = cf.w_lin_z * v_b[..., 2] ** 2 + cf.w_ang_xy * (w_b[..., :2] ** 2).sum(-1)
        g_torso = -rot[..., 0, 2, :]
        r_up = torch.exp(-(g_torso[..., :2] ** 2).sum(-1) / cf.std_upright**2)
        stand = (speed < cf.walking_threshold).float().unsqueeze(-1)
        run = (speed >= cf.running_threshold).float().unsqueeze(-1)
        std = self.std["std_standing"] * stand + self.std["std_running"] * run + self.std["std_walking"] * (1 - stand - run)
        r_pose = torch.exp(-(((q - self.default_q) / std) ** 2).mean(-1)) * stand.squeeze(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r_rate = ((a - a_prev) ** 2).sum(-1)
        vxy = fv[..., :2].norm(dim=-1)
        dc = torch.relu(cf.clearance_target - K["gap_min"])
        r_clear = (dc * vxy).sum(-1) * active
        r_slip = ((fv[..., :2] ** 2).sum(-1) * c).sum(-1) * active
        r_bav = (w[..., :2] ** 2).sum(-1)
        cl, cr = c[..., 0], c[..., 1]
        walking = (speed > 0.2).float()
        r_gait = walking * (cl * (1 - cr) + cr * (1 - cl) - 2 * (1 - cl) * (1 - cr))
        # sole-support term (also stair edges) + smooth support field (gaps, wide gradient band)
        u = K["unsup"].sum(-1)
        r_foothold = (c * u).sum(-1)
        r_still = stand.squeeze(-1) * (1 - c).sum(-1)
        dev_abs = (q - self.default_q).abs()
        r_dev = (cf.w_dev_waist * (dev_abs * self.m_waist).sum(-1) + cf.w_dev_hip * (dev_abs * self.m_hip).sum(-1)
                 + cf.w_dev_arms * (dev_abs * self.m_arms).sum(-1) + cf.w_dev_arm_side * (dev_abs * self.m_arm_side).sum(-1))
        r_edge = (c * K["edge"]).sum(-1)
        r_flat = (c * K["flat"]).sum(-1)
        sp_b = torch.sqrt((v_b[..., :2] ** 2).sum(-1) + 1e-6)  # safe norm (finite gradient at rest)
        r_over = torch.relu(sp_b - cmd[..., :2].detach().norm(dim=-1) - 0.1) ** 2
        r_dev = r_dev + cf.w_edge * r_edge + cf.w_flat * r_flat + cf.w_overspeed * r_over
        r_troll = g_torso[..., 1] ** 2
        r_troll_rate = (w_b[..., 0] + (qd * self.m_waist_roll).sum(-1)) ** 2
        r_dev = r_dev + cf.w_torso_roll * r_troll + cf.w_torso_roll_rate * r_troll_rate
        r_impact = (torch.relu(K["ff"] - cf.impact_n) / 100.0).sum(-1)  # sole forces (N) of the world model
        r = (cf.w_impact * r_impact + cf.w_lin * r_lin + cf.w_ang * r_ang + cf.w_upright * r_up + cf.w_pose * r_pose + cf.w_limits * r_lim
             + cf.w_action_rate * r_rate + cf.w_clearance * r_clear + cf.w_slip * r_slip + cf.w_body_ang_vel * r_bav
             + cf.w_gait * r_gait + cf.w_foothold * r_foothold + cf.w_still * r_still + r_dev + r_extra) * dt  # fmt: skip
        return r

    def lava_hit(self, fp, clear) -> torch.Tensor:
        """A foot (site over a lava cell) within lava_clear of the floor under it: it stepped into a stones gap."""
        lava = self.terrain.hm.query_lava(fp[..., :2])
        return ((lava > 0.5) & (clear < self.cfg.lava_clear)).any(-1)

    def soft_termination(self, S, K, tz: float = 0.02, tt: float = 0.03) -> torch.Tensor:
        R = S[1]
        alive = torch.sigmoid((R[..., 2, 2] - math.cos(self.cfg.term_angle)) / tt) * torch.sigmoid(
            (K["hb"] - self.cfg.term_height) / tz)
        if self.cfg.term_hole > 0:
            alive = alive * torch.sigmoid((K["hole"] - self.cfg.term_hole) / tz)
        if self.cfg.lava_clear > 0:  # each foot: lava under it x close to the floor
            lava = self.terrain.hm.query_lava(K["fp"][..., :2])
            alive = alive * (1.0 - lava * torch.sigmoid((self.cfg.lava_clear - K["clear"]) / tz)).prod(-1)
        return 1 - alive

    def critic_features(self, S, K, c, cmd, xi_n) -> torch.Tensor:
        p, R = S[0], S[1]
        H = yaw_mat(R)
        feet = (H.transpose(-1, -2).unsqueeze(-3) @ (K["fp"] - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1).flatten(-2)
        scan = self.terrain.critic_scan(p, R)
        return torch.cat((invariant_features(S), K["tf"], c, feet, cmd, xi_n, scan), -1)
