"""Differentiable MDP heads shared by the simulator and imagination: state, rewards and terminations.

State ``S = (p, R, v, w, q, qd)``: base position, rotation, linear and angular velocity, joint positions and
velocities. Contact-dependent rewards use the world model's foot-contact probabilities and contact forces, the terrain
under the feet and the clearances of the body's collision spheres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .interface import CMD_DIM, JointMap, base_yaw, goal_features, policy_proprio, system_state, wrap_angle
from .kinematics import Kinematics
from .rotations import mat_to_6d, quat_to_mat, yaw_mat
from ..constants import EFFORT_LIMIT, FOOT_RADIUS, KD, KP, NUM_FEET, NUM_J
from .terrain import TerrainFeatures

INV_DIM = 6 + 3 + 3 + NUM_J + NUM_J  # height comes from the terrain features


@dataclass
class RewardCfg:
    """AME2's goal-reaching reward (Zhang et al. 2026, Table I), differentiable, plus self-collision, goal-facing,
    lateral-velocity, foot-impact and power terms; position-goal task of Rudin et al. 2022: reach the target position
    and heading by the end of the episode (remaining time t*). Weights are per second (scaled by dt)."""
    w_term: float = -500.0  # termination
    qd_lim: float = 7.5  # ANYdrive velocity limit (rad/s)
    w_self: float = -2.0  # sum over sphere / foot pairs of different legs of clamp((self_margin - gap) / self_margin, 0, 1)
    self_margin: float = 0.05
    w_face: float = 2.0  # face the goal while away from it: (1 - near) cos(yaw - bearing to the goal)
    w_lat: float = -1.0  # lateral base velocity squared (body frame)
    # foot impact (the world model's contact forces, K["ff"]): anti-stomping
    w_impact: float = -3.0  # sum over feet of max(|F_foot| - impact_n, 0) / 100 N
    impact_n: float = 350.0
    w_power: float = -0.002  # mechanical power: sum_j |tau_j qd_j| (W), tau from the PD law (constants.KP / KD)
    term_angle: float = 1.4
    term_height: float = 0.25  # base above the terrain under it
    term_floor: float = -2.5  # absolute base height (m) below which the robot has fallen into a floorless gap
    term_clear: float = 0.0  # a forbidden collision sphere closer than this to the terrain (m)


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:13 + NUM_J], x[..., 13 + NUM_J:13 + 2 * NUM_J]


def invariant_features(S) -> torch.Tensor:
    """Heading/position-invariant features: tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (mat_to_6d(H.transpose(-1, -2) @ R), (Rt @ v.unsqueeze(-1)).squeeze(-1), (Rt @ w.unsqueeze(-1)).squeeze(-1),
         q, qd), -1)  # fmt: skip


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Dynamics-relevant context only; the encoder bias (xi[4:4+NUM_J]) only affects observations."""
    e = xi[..., 4 + NUM_J :]
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:4] / 0.05, e[..., 0:1] / 0.15 - 1, e[..., 1:2] / 0.075 - 1,
                      (e[..., 2:3] - 1.1) / 0.4, e[..., 3:4] / 2 - 1, e[..., 4:5] / 5), -1)  # fmt: skip


XI_N_DIM = 9


class MDP:
    def __init__(self, kin: Kinematics, terrain: TerrainFeatures, default_q: torch.Tensor, joint_names: list[str],
                 soft_lo: torch.Tensor, soft_hi: torch.Tensor, action_scale: torch.Tensor,
                 cfg: RewardCfg = RewardCfg()):
        self.kin, self.terrain, self.cfg = kin, terrain, cfg
        self.default_q = default_q
        self.action_scale = action_scale
        self.soft_lo, self.soft_hi = soft_lo, soft_hi
        dev = default_q.device
        self.jmap = JointMap(joint_names, dev)
        self.q0_isaac = self.jmap.to_isaac(default_q)
        # leg-leg self-collision pairs (collision spheres + the feet)
        legs = [n[:2] if n[:2] in ("LF", "RF", "LH", "RH") else "base" for n in kin.col_names] + ["LF", "RF", "LH", "RH"]
        ii, jj = [], []
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                if legs[i] != legs[j] and "base" not in (legs[i], legs[j]):
                    ii.append(i), jj.append(j)
        rad = torch.cat((kin.col_radius, torch.full((4,), FOOT_RADIUS, device=dev)))
        self.self_pairs = (torch.tensor(ii, device=dev), torch.tensor(jj, device=dev), rad[ii] + rad[jj])
        self.critic_dim = INV_DIM + terrain.wm_dim + NUM_FEET + 3 * NUM_FEET + CMD_DIM + XI_N_DIM + terrain.critic_dim

    def _fk(self, S):
        p, R, v, w, q, qd = S
        out = self.kin.forward(p, R, v, w, q, qd, collision=True)
        fp = out[0] - torch.tensor((0.0, 0.0, FOOT_RADIUS), device=p.device)  # contact point below the sphere centre
        return (fp,) + tuple(out[1:])

    def context(self, S) -> dict:
        p, R = S[0], S[1]
        fp, fv, rot, bpos, cp = self._fk(S)
        tf, clear, rough, unsup, clr, geo = self.terrain.wm_features(p, R, fp, cp)
        K = {"fp": fp, "fv": fv, "rot": rot, "tf": tf, "clear": clear, "rough": rough, "unsup": unsup, "hb": tf[..., 0],
             "torso_p": bpos[..., 0, :], "torso_R": rot[..., 0, :, :], "cp": cp, "clr": clr, "geo": self._geo(S, geo)}
        # gaps between spheres / feet of different legs (..., n_pairs)
        ii, jj, rr = self.self_pairs
        pts = torch.cat((cp, fp + torch.tensor((0.0, 0.0, FOOT_RADIUS), device=p.device)), -2)
        K["self_gap"] = torch.sqrt(((pts[..., ii, :] - pts[..., jj, :]) ** 2).sum(-1) + 1e-9) - rr
        return K

    def wm_terrain(self, S) -> torch.Tensor:
        """Only the world model's terrain input ``context(S)["tf"]``."""
        p, R = S[0], S[1]
        fp, fv, rot, bpos, cp = self._fk(S)
        return self.terrain.wm_features(p, R, fp, cp)[0]

    def wm_inputs(self, S):
        """World-model inputs from the state: terrain features and the exact contact geometry."""
        p, R = S[0], S[1]
        fp, fv, rot, bpos, cp = self._fk(S)
        out = self.terrain.wm_features(p, R, fp, cp)
        return out[0], self._geo(S, out[5])

    def _geo(self, S, geo) -> dict:
        """+ the hinge frames (columns of J) and the joint mask of each contact body, for J^T F."""
        ax, org = self.kin.joint_frames(S[0], S[1], S[4])
        return dict(geo, ax=ax, org=org)

    def min_clearance(self, S) -> torch.Tensor:
        """Smallest clearance of a forbidden collision sphere to the terrain (...,) (non-forbidden ones offset by 1 m)."""
        return (self.terrain.clearance(self._fk(S)[4]) + self.terrain.term_offset).amin(-1)

    def proprio(self, S, a_prev_mj, cmd, xi, noise: bool = True) -> torch.Tensor:
        """Policy proprioception: the sim encoder bias xi[4:4+NUM_J] corrupts the joint positions."""
        return policy_proprio(S, self.jmap.to_isaac(a_prev_mj), cmd, self.q0_isaac, self.jmap,
                              bias_mj=xi[..., 4 : 4 + NUM_J], noise=noise)

    def system_state(self, S, cmd) -> torch.Tensor:
        return system_state(S, cmd, self.q0_isaac, self.jmap)

    def joint_target(self, a_mj: torch.Tensor) -> torch.Tensor:
        """Joint position setpoint, the action-convention-free world-model input."""
        return self.default_q + self.action_scale * a_mj

    def reward(self, S, K, c, a, a_prev, cmd, dt: float, terms: bool = False, prev=None, d=None):
        """Per-step reward (scaled by dt). ``cmd``: goal (..., 4) = target x, y, heading, remaining time;
        ``prev``: (v, w, qd, foot velocities) one step earlier (acceleration terms); ``d``: termination probability."""
        out = self.reward_ame2(S, K, c, a, a_prev, cmd, dt, terms, prev, d)
        r, tm = out if terms else (out, None)
        x, xt = self.shaping(S, K, cmd)
        r = r + x * dt
        if terms:
            return r, dict(tm, **xt)
        return r

    def shaping(self, S, K, cmd):
        """Self-collision proximity, facing the goal, lateral velocity and foot impacts (per second)."""
        cf = self.cfg
        p, R, v = S[0], S[1], S[2]
        zero = torch.zeros_like(p[..., 0])
        m = cf.self_margin
        r_self = ((m - K["self_gap"]) / m).clamp(0.0, 1.0).sum(-1)
        d = cmd[..., :2] - p[..., :2]
        dist = torch.sqrt((d**2).sum(-1) + 1e-8)
        far = torch.sigmoid((dist - 0.5) / 0.05)
        r_face = far * torch.cos(base_yaw(R) - torch.atan2(d[..., 1], d[..., 0]))
        v_b = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        r_lat = v_b[..., 1] ** 2
        r_impact = zero
        if "ff" in K:  # foot force magnitudes (N) predicted by the world model (imagination only)
            r_impact = (torch.relu(K["ff"] - cf.impact_n) / 100.0).sum(-1)
        x = cf.w_self * r_self + cf.w_face * r_face + cf.w_lat * r_lat + cf.w_impact * r_impact
        return x, {"self": r_self, "face": r_face, "lat": r_lat, "impact": r_impact}

    def reward_ame2(self, S, K, c, a, a_prev, cmd, dt: float, terms: bool = False, prev=None, d=None):
        """AME2's goal-reaching reward (Table I, Eqs. 1-5), differentiable; weights per second (x dt).

        Undesired events (-1 each): spinning (|w_z| > 2), leaping on flat terrain (all feet off the ground, < 30 cm
        elevation spread under the feet), non-foot contact (any tolerated / forbidden collision sphere within its
        margin), foot slippage (moving > 0.1 m/s in contact). Not included: stumbling, contact-switch counting and link
        contact forces; self-collision is the separate proximity term of ``shaping``.
        """
        p, R, v, w, q, qd = S
        sig = torch.sigmoid
        Rt = R.transpose(-1, -2)
        v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
        w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
        g_b = -R[..., 2, :]
        t_left = cmd[..., 3]
        d2 = ((p[..., :2] - cmd[..., :2]) ** 2).sum(-1)
        dxy = torch.sqrt(d2 + 1e-8)
        dyaw = wrap_angle(base_yaw(R) - cmd[..., 2]).abs()
        near = sig((0.5 - dxy) / 0.03)
        r_pos = (1 / (1 + 0.25 * d2)) * (t_left < 4.0).float() / 4.0
        r_head = (1 / (1 + dyaw**2)) * (t_left < 2.0).float() / 2.0 * near
        speed = torch.sqrt((v_b[..., :2] ** 2).sum(-1) + 1e-8)
        to_goal = cmd[..., :2] - p[..., :2]
        speed_w = torch.sqrt((v[..., :2] ** 2).sum(-1) + 1e-8)
        cos = (v[..., :2] * to_goal).sum(-1) / (speed_w * dxy)
        moving = sig((cos - 0.5) / 0.05) * sig((speed - 0.3) / 0.03) * sig((2.0 - speed) / 0.03)
        r_move = near + (1 - near) * moving
        d_foot = (1 - c).mean(-1)
        d_g = 1 - g_b[..., 2] ** 2
        d_q = (q - self.default_q).abs().mean(-1)
        r_stand = near * sig((0.5 - dyaw) / 0.05) * torch.exp(-(d_foot + d_g + d_q + dxy) / 4)
        # undesired events
        e_spin = sig((w_b[..., 2].abs() - 2.0) / 0.1)
        h_feet = K["fp"][..., 2] - K["clear"]  # terrain height under each foot
        spread = h_feet.max(-1).values - h_feet.min(-1).values
        e_leap = torch.prod(1 - c, -1) * sig((0.3 - spread) / 0.03)
        m = self.terrain.col_margin
        prox = ((m - K["clr"]) / m).clamp(0.0, 1.0) * (self.terrain.col_weight > 0).float()
        e_contact = prox.amax(-1)
        fvn = torch.sqrt((K["fv"][..., :2] ** 2).sum(-1) + 1e-8)
        e_slip = (c * sig((fvn - 0.1) / 0.03)).sum(-1)
        r_events = e_spin + e_leap + e_contact + e_slip
        r_roll = w_b[..., 0] ** 2
        tau_u = KP * (self.joint_target(a) - q) - KD * qd
        tau = tau_u.clamp(-EFFORT_LIMIT, EFFORT_LIMIT)
        zero = torch.zeros_like(dxy)
        qdd, r_link_acc = zero, zero
        if prev is not None:
            v0, w0, qd0, fv0 = prev
            qdd = (((qd - qd0) / dt) ** 2).sum(-1)
            r_link_acc = torch.sqrt((((v - v0) / dt) ** 2).sum(-1) + 1e-8) + torch.sqrt(
                (((K["fv"] - fv0) / dt) ** 2).sum(-1) + 1e-8).sum(-1)
        r_joint = (qd**2).sum(-1) + 0.01 * (tau**2).sum(-1) + 0.001 * qdd
        r_rate = ((a - a_prev) ** 2).sum(-1)
        span = self.soft_hi - self.soft_lo
        r_qlim = (torch.relu(q - (self.soft_hi - 0.025 * span)) + torch.relu((self.soft_lo + 0.025 * span) - q)).sum(-1)
        r_qdlim = torch.relu(qd.abs() - 0.9 * self.cfg.qd_lim).sum(-1)
        r_taulim = torch.relu(tau_u.abs() - 0.8 * EFFORT_LIMIT).sum(-1)
        r_term = d if d is not None else zero
        r_power = torch.sqrt((tau * qd) ** 2 + 1e-6).sum(-1)
        r = (self.cfg.w_power * r_power + 100.0 * r_pos + 50.0 * r_head + 5.0 * r_move + 5.0 * r_stand + self.cfg.w_term * r_term - 1.0 * r_events
             - 0.1 * r_roll - 0.001 * r_joint - 0.01 * r_rate - 0.001 * r_link_acc - 1000.0 * r_qlim - 1.0 * r_qdlim
             - 1.0 * r_taulim) * dt  # fmt: skip
        if terms:
            return r, {"pos": r_pos, "head": r_head, "move": r_move, "stand": r_stand, "term": r_term,
                       "events": r_events, "roll": r_roll, "joint": r_joint, "rate": r_rate, "link_acc": r_link_acc,
                       "qlim": r_qlim, "qdlim": r_qdlim, "taulim": r_taulim, "power": r_power}  # fmt: skip
        return r

    def soft_termination(self, S, K, tz: float = 0.02, tt: float = 0.03, tc: float = 0.005) -> torch.Tensor:
        """Differentiable termination probability: tilt, base height above the terrain, fall into a floorless gap,
        forbidden collision sphere at the terrain."""
        R = S[1]
        alive = torch.sigmoid((R[..., 2, 2] - math.cos(self.cfg.term_angle)) / tt) * torch.sigmoid(
            (K["hb"] - self.cfg.term_height) / tz)
        alive = alive * torch.sigmoid((S[0][..., 2] - self.cfg.term_floor) / tz)
        # argmin + gather (compiled amin gave NaN gradients)
        c = K["clr"] + self.terrain.term_offset
        cmin = c.gather(-1, c.detach().argmin(-1, keepdim=True)).squeeze(-1)
        alive = alive * torch.sigmoid((cmin - self.cfg.term_clear) / tc)
        return 1 - alive

    def critic_features(self, S, K, c, cmd, xi_n) -> torch.Tensor:
        p, R = S[0], S[1]
        H = yaw_mat(R)
        feet = (H.transpose(-1, -2).unsqueeze(-3) @ (K["fp"] - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1).flatten(-2)
        scan = self.terrain.critic_scan(p, R)
        return torch.cat((invariant_features(S), K["tf"], c, feet, goal_features(S, cmd), xi_n, scan), -1)
