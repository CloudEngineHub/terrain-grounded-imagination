"""Observation and action conventions of the ANYmal-D env (Isaac Lab's ANYmal-D rough task, FPM foothold scan).

* joint order: Isaac Lab breadth-first (``ISAAC_JOINTS``); action: ``q_target = q0 + 0.5 a`` (Isaac Lab default);
* policy observation (158): base_lin_vel, 0.2 base_ang_vel, projected_gravity, goal features (5), joint_pos - q0,
  0.05 joint_vel, last_action, foothold_scan (4 x (2 + 5 x 5)); noise before scaling;
* FPM input ``system_state`` (38) -> next footholds (x, y) of the four feet (LF, RF, LH, RH) in the base frame;
* foothold_scan: 5 x 5 nodes around each foothold on the base's yaw-aligned 1.6 x 1.0 m grid (0.05 m, the hires
  height map), ``clip(base_z - terrain_z - 0.5, -1, 1)``.
"""

from __future__ import annotations

import torch

from .rotations import yaw_mat

ISAAC_JOINTS = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]  # fmt: skip

ACTION_CLIP = 10.0

NJ = len(ISAAC_JOINTS)
N_FEET = 4
# goal command: world target x, y, target heading, remaining time (s)
CMD_DIM = 5  # its observation: target x, y in the yaw frame (norm clipped at GOAL_CLIP), sin / cos heading error, time
GOAL_CLIP = 5.0
PROPRIO_DIM = 3 + 3 + 3 + CMD_DIM + NJ + NJ + NJ
SYSTEM_STATE_DIM = 3 + 3 + 3 + CMD_DIM + NJ + NJ
PATCH_N = 5
SCAN_RES = 0.05
SCAN_SIZE = (1.6, 1.0)
SCAN_OFFSET = 0.5
FOOTHOLD_SCAN_DIM = N_FEET * (2 + PATCH_N**2)
# nominal stance at the default pose (LF, RF, LH, RH), base frame
BOOTSTRAP_FOOTHOLDS = ((0.5, 0.31), (0.5, -0.31), (-0.5, 0.31), (-0.5, -0.31))

# Isaac observation noise (uniform half-widths, applied before scaling) and scales
LIN_VEL_NOISE, ANG_VEL_NOISE, GRAV_NOISE, QPOS_NOISE, QVEL_NOISE, SCAN_NOISE = 0.15, 0.2, 0.05, 0.01, 1.5, 0.1
ANG_VEL_SCALE, QVEL_SCALE = 0.2, 0.05
XY_DRIFT, Z_DRIFT = 0.05, 0.05


class JointMap:
    """Permutations between the simulator's (MJCF) joint order and Isaac Lab's order."""

    def __init__(self, mj_joints: list[str], device: str):
        assert sorted(mj_joints) == sorted(ISAAC_JOINTS)
        self.m2i = torch.tensor([mj_joints.index(n) for n in ISAAC_JOINTS], device=device)  # x_isaac = x_mj[m2i]
        self.i2m = torch.tensor([ISAAC_JOINTS.index(n) for n in mj_joints], device=device)  # x_mj = x_isaac[i2m]

    def to_isaac(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.m2i]

    def to_mj(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.i2m]


def _u(x: torch.Tensor, h: float) -> torch.Tensor:
    return (2 * torch.rand_like(x) - 1) * h


def base_quantities(S):
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
    w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
    g_b = -R[..., 2, :]
    return v_b, w_b, g_b


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def base_yaw(R: torch.Tensor) -> torch.Tensor:
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def goal_features(S, cmd: torch.Tensor) -> torch.Tensor:
    """(..., CMD_DIM) from the goal command (..., 4): target in the yaw frame, heading error (sin, cos), time left."""
    p, R = S[0], S[1]
    yaw = base_yaw(R)
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx, dy = cmd[..., 0] - p[..., 0], cmd[..., 1] - p[..., 1]
    xb, yb = c * dx + s * dy, -s * dx + c * dy
    n = torch.sqrt(xb**2 + yb**2 + 1e-8)
    k = (GOAL_CLIP / n).clamp(max=1.0)
    e = cmd[..., 2] - yaw
    return torch.stack((xb * k, yb * k, torch.sin(e), torch.cos(e), cmd[..., 3]), -1)


def policy_proprio(S, a_prev_isaac, cmd, q0_isaac, jmap: JointMap, bias_mj=None, noise: bool = True) -> torch.Tensor:
    """Proprioceptive part of the policy observation (Isaac order). ``bias_mj``: encoder bias (sim DR)."""
    q, qd = S[4], S[5]
    v_b, w_b, g_b = base_quantities(S)
    q_rel = jmap.to_isaac(q if bias_mj is None else q + bias_mj) - q0_isaac
    qd_i = jmap.to_isaac(qd)
    if noise:
        spike = (torch.rand_like(v_b[..., :1]) < 1e-4).float()
        v_b = v_b + _u(v_b, LIN_VEL_NOISE) + spike * _u(v_b, 5.0)
        w_b = w_b + _u(w_b, ANG_VEL_NOISE) + (torch.rand_like(w_b[..., :1]) < 1e-4).float() * _u(w_b, 10.0)
        g_b = g_b + _u(g_b, GRAV_NOISE)
        q_rel = q_rel + _u(q_rel, QPOS_NOISE)
        qd_i = qd_i + _u(qd_i, QVEL_NOISE)
    return torch.cat((v_b, ANG_VEL_SCALE * w_b, g_b, goal_features(S, cmd), q_rel, QVEL_SCALE * qd_i,
                      a_prev_isaac.clamp(-ACTION_CLIP, ACTION_CLIP)), -1)  # fmt: skip


def system_state(S, cmd, q0_isaac, jmap: JointMap) -> torch.Tensor:
    """FPM input (38): base_lin_vel, base_ang_vel, projected_gravity, goal features, joint_pos_rel, joint_vel."""
    v_b, w_b, g_b = base_quantities(S)
    return torch.cat((v_b, w_b, g_b, goal_features(S, cmd), jmap.to_isaac(S[4]) - q0_isaac, jmap.to_isaac(S[5])), -1)


class FootholdScan:
    """The FPM ``foothold_scan`` observation on top of a global heightmap (stand-in for a base-mounted ray caster)."""

    def __init__(self, hm, device: str):
        self.hm = hm
        self.nx = int(SCAN_SIZE[0] / SCAN_RES) + 1
        self.ny = int(SCAN_SIZE[1] / SCAN_RES) + 1
        k = torch.arange(-(PATCH_N // 2), PATCH_N // 2 + 1, device=device)
        ox, oy = torch.meshgrid(k, k, indexing="ij")
        self.off_x, self.off_y = ox.flatten(), oy.flatten()

    def __call__(self, p, R, torso_p, torso_R, footholds_root, xy_drift=None, z_drift=None, noise: bool = True):
        """footholds_root: (..., n_feet, 2) in the base frame. Returns (..., n_feet * (2 + PATCH_N^2))."""
        fb = torch.cat((footholds_root, torch.zeros_like(footholds_root[..., :1])), -1)
        feet_w = p.unsqueeze(-2) + (R.unsqueeze(-3) @ fb.unsqueeze(-1)).squeeze(-1)
        rel = feet_w - torso_p.unsqueeze(-2)
        feet_s = (torso_R.transpose(-1, -2).unsqueeze(-3) @ rel.unsqueeze(-1)).squeeze(-1)  # full sensor rotation
        if xy_drift is not None:
            feet_s = torch.cat((feet_s[..., :2] + xy_drift.unsqueeze(-2), feet_s[..., 2:]), -1)
        ax = torch.round((feet_s[..., 0] + SCAN_SIZE[0] / 2) / SCAN_RES).long()
        ay = torch.round((feet_s[..., 1] + SCAN_SIZE[1] / 2) / SCAN_RES).long()
        ix = (ax.unsqueeze(-1) + self.off_x).clamp(0, self.nx - 1)
        iy = (ay.unsqueeze(-1) + self.off_y).clamp(0, self.ny - 1)
        gx = ix.float() * SCAN_RES - SCAN_SIZE[0] / 2
        gy = iy.float() * SCAN_RES - SCAN_SIZE[1] / 2
        Y = yaw_mat(torso_R)[..., :2, :2].unsqueeze(-3).unsqueeze(-3)  # yaw-aligned grid
        pts = torso_p[..., None, None, :2] + (Y @ torch.stack((gx, gy), -1).unsqueeze(-1)).squeeze(-1)
        hz = self.hm.query_nearest(pts)
        h = torso_p[..., 2, None, None] - hz - SCAN_OFFSET
        if z_drift is not None:  # drift before noise and clipping
            h = h + z_drift[..., None, None]
        out = torch.cat((footholds_root, h), -1)  # (..., n_feet, 2 + PATCH_N^2)
        if noise:
            out = out + _u(out, SCAN_NOISE)
        return out.clamp(-1.0, 1.0).flatten(-2)


# Left/right mirror of ANYmal (Isaac joint order): LF<->RF, LH<->RH; HAA flips sign, HFE and KFE keep it.
JOINT_PERM = [ISAAC_JOINTS.index(n.replace("L", "#").replace("R", "L").replace("#", "R")) for n in ISAAC_JOINTS]
JOINT_SIGN = [-1.0 if n.endswith("HAA") else 1.0 for n in ISAAC_JOINTS]
FOOT_PERM = [1, 0, 3, 2]  # (LF, RF, LH, RH) -> (RF, LF, RH, LH)


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
    return x[..., JOINT_PERM] * torch.tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)


def mirror_proprio(o: torch.Tensor) -> torch.Tensor:
    """Mirror the proprioceptive policy observation left/right."""
    s3 = lambda x, s: x * torch.tensor(s, device=x.device, dtype=x.dtype)  # noqa: E731
    a, b = 9 + CMD_DIM, 9 + CMD_DIM + NJ
    return torch.cat((s3(o[..., 0:3], (1.0, -1.0, 1.0)), s3(o[..., 3:6], (-1.0, 1.0, -1.0)), s3(o[..., 6:9], (1.0, -1.0, 1.0)),
                      s3(o[..., 9:a], (1.0, -1.0, -1.0, 1.0, 1.0)), mirror_joints(o[..., a:b]), mirror_joints(o[..., b:b + NJ]),
                      mirror_joints(o[..., b + NJ:b + 2 * NJ])), -1)  # fmt: skip


def mirror_scan(e: torch.Tensor) -> torch.Tensor:
    """Mirror the foothold scan: swap left/right feet, negate foothold y, flip each patch along y."""
    k = PATCH_N
    stride = 2 + k * k
    feet = [e[..., i * stride:(i + 1) * stride] for i in range(N_FEET)]
    out = []
    for i in FOOT_PERM:
        src = feet[i]
        patch = src[..., 2:].reshape(*src.shape[:-1], k, k).flip(-1).reshape(*src.shape[:-1], k * k)
        out.append(torch.cat((src[..., 0:1], -src[..., 1:2], patch), -1))
    return torch.cat(out, -1)
