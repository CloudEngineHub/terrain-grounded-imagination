"""Contact-force world model, policy, and critic."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import NUM_FEET, NUM_J
from ..mdp.mdp import INV_DIM, XI_N_DIM, invariant_features
from ..mdp.rotations import so3_exp, so3_log, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        # in place: buffers keep their addresses (CUDA-graph replays read them by address)
        self.var.copy_((self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot)
        self.mean += delta * n / tot
        self.count.copy_(tot)

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


DYN_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic s_{t+1} = f(s_t, a_t, xi) on frame-local increments: inputs, increment targets, state error and their
    normalizers (the network is ``ImpulseWorldModel``'s)."""

    def __init__(self, terrain_dim: int, hidden=(1024, 1024, 1024)):
        super().__init__()
        self.in_dim = INV_DIM + terrain_dim + 2 * NUM_J + XI_N_DIM
        self.net = mlp(self.in_dim, list(hidden), DYN_OUT + NUM_FEET, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(DYN_OUT, eps=1e-4)

    def inputs(self, S, tf, a, a_prev, xi_n):
        """tf: terrain features; a_prev: previous action (matters under actuation delay)."""
        return torch.cat((invariant_features(S), tf, a, a_prev, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            ((H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ R1),
             (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1), (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1), q1 - q, qd1 - qd), -1)  # fmt: skip

    @staticmethod
    def state_error(S_hat, S):
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            ((H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ Rh),
             (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1), (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1), qh - q, qdh - qd), -1)  # fmt: skip


class ImpulseWorldModel(WorldModel):
    """Contact-force world model. The next state is the free motion plus the effect of the
    contact forces through the exact contact Jacobian,

        s_{t+1} = s_t + f(x) + B(x) J(s_t)^T F,   F_b = p_b m_b n_b + f_t,b   for every contact body b,

    where only f (free motion), B (effective inverse inertia x dt, in the normalized increment space) and each body's
    load probability p, normal force magnitude m and friction direction are learned. Where, and in which direction, a
    body can be pushed is exact: the gap phi_b to the 3D field gates p (no force away from the surface), the field's
    normal n_b sets the normal direction, the tangential force stays inside the friction cone (|f_t| <= mu p m), and
    J^T F is computed from the kinematics at the contact point. Labels: the simulator's net contact force per body."""

    GATE_EPS, GATE_TAU = 0.04, 0.01  # gap (m) below which a body can be loaded (one 20 ms step of approach), softness
    F0, MU = 100.0, 1.0  # force unit (N), friction-cone coefficient
    Q_SCALE = (500.0, 100.0, 50.0)  # generalized force units: base force (N), base moment (N m), joint torque (N m)

    def __init__(self, terrain_dim: int, hidden, joint_mask: torch.Tensor):
        super().__init__(terrain_dim, hidden)
        self.nb = joint_mask.shape[0]
        self.nq = 6 + joint_mask.shape[1]
        n_out = DYN_OUT + 4 * self.nb + DYN_OUT * self.nq
        self.net = mlp(self.in_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("jmask", joint_mask.float())

    def step_full(self, S, tf, a, a_prev, xi_n, geo):
        """Returns (S1, foot logits, contact forces F (..., nb, 3) world, load probabilities (..., nb))."""
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, tf, a, a_prev, xi_n)))
        y_free = y[..., :DYN_OUT]
        yc = y[..., DYN_OUT : DYN_OUT + 4 * self.nb].unflatten(-1, (self.nb, 4))
        Bm = y[..., DYN_OUT + 4 * self.nb :].unflatten(-1, (DYN_OUT, self.nq))
        H = yaw_mat(R)
        phi, n, x = geo["phi"], geo["n"], geo["x"]
        gate = torch.sigmoid((self.GATE_EPS - phi) / self.GATE_TAU)
        p_load = gate * torch.sigmoid(yc[..., 0])
        lam = p_load * nn.functional.softplus(yc[..., 1]) * self.F0
        t3 = (H.unsqueeze(-3) @ nn.functional.pad(yc[..., 2:4], (0, 1)).unsqueeze(-1)).squeeze(-1)
        t_perp = t3 - (t3 * n).sum(-1, keepdim=True) * n
        t_norm = torch.sqrt((t_perp**2).sum(-1, keepdim=True) + 1e-8)
        F = lam.unsqueeze(-1) * (n + self.MU * torch.tanh(t_norm) * t_perp / t_norm)
        # generalized contact force J^T F: base force and moment (base frame), joint torques
        f_w = F.sum(-2)
        m_w = torch.cross(x - p.unsqueeze(-2), F, dim=-1).sum(-2)
        lever = x.unsqueeze(-2) - geo["org"].unsqueeze(-3)  # (..., nb, nj, 3)
        tau_b = (geo["ax"].unsqueeze(-3) * torch.cross(lever, F.unsqueeze(-2).expand_as(lever), dim=-1)).sum(-1)
        tau = (tau_b * self.jmask).sum(-2)
        Rt = R.transpose(-1, -2)
        fs, ms, ts = self.Q_SCALE
        Q = torch.cat(((Rt @ f_w.unsqueeze(-1)).squeeze(-1) / fs, (Rt @ m_w.unsqueeze(-1)).squeeze(-1) / ms, tau / ts), -1)
        y_dyn = y_free + (Bm @ Q.unsqueeze(-1)).squeeze(-1)
        d = y_dyn * self.out_norm.std() + self.out_norm.mean
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        pf = p_load[..., :NUM_FEET].clamp(1e-5, 1 - 1e-5)
        return (p1, R1, v1, w1, q + dq, qd + dqd), torch.log(pf) - torch.log1p(-pf), F, p_load


class Policy(nn.Module):
    """Deployable MLP over a short observation history (newest last), with built-in normalization."""

    def __init__(self, obs_dim: int, hist: int = 5, hidden=(512, 256, 128), extra_dim: int = 0):
        super().__init__()
        self.hist, self.obs_dim, self.extra_dim = hist, obs_dim, extra_dim
        self.net = mlp(obs_dim * hist + extra_dim, list(hidden), NUM_J)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.extra_norm = RunningNorm(max(extra_dim, 1))

    def forward(self, obs_hist: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        x = self.obs_norm(obs_hist).flatten(-2)
        if self.extra_dim:
            x = torch.cat((x, self.extra_norm(extra)), -1)
        return run_mlp(self.net, x)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256)):
        super().__init__()
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
