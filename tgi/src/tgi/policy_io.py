"""Closed-loop policy execution in the simulator and checkpoint loading."""

from __future__ import annotations

import torch

from .foot_pred import FootholdPredictor
from .mdp import MDP, G1Kinematics, RewardCfg, TerrainFeatures
from .mdp.interface import BOOTSTRAP_FOOTHOLDS, FOOTHOLD_SCAN_DIM, PROPRIO_DIM, XY_DRIFT, Z_DRIFT, FootholdScan
from .world_model import Policy


def build_mdp(sim, device: str, reward_cfg: RewardCfg | None = None, field_window: float = 0.5,
              field_hole: float = 0.25, hm_res: float = 0.04) -> MDP:
    """Floor mesh -> heightmap (+ lava, support and edge fields); MJCF kinematics."""
    hm = sim.heightmap(device, res=hm_res)
    hm.lava = sim.lava_mask(hm).to(hm.h.device)
    print(f"lava floors: {float(hm.lava.mean()):.3f} of the heightmap cells", flush=True)
    hm.build_support_field(window=field_window, hole_depth=field_hole)
    hm.build_edge_field()
    lo, hi = sim.soft_limits
    kin = G1Kinematics(sim.kin_model, sim.joint_names, device)
    terrain = TerrainFeatures(hm, device)
    mdp = MDP(kin, terrain, sim.default_q, sim.joint_names, lo, hi, sim.action_scale, reward_cfg or RewardCfg())
    mdp.scan = FootholdScan(hm, device)
    return mdp


def bootstrap_footholds(n: int, device: str) -> torch.Tensor:
    return torch.tensor(BOOTSTRAP_FOOTHOLDS, device=device).expand(n, 2, 2)


def perceive(mdp: MDP, S, K, fh, xy_drift=None, z_drift=None, noise: bool = True):
    """The policy's exteroception from a state: the foothold scan around the footholds ``fh``."""
    return mdp.scan(S[0], S[1], K["torso_p"], K["torso_R"], fh, xy_drift, z_drift, noise)


def make_policy(device: str) -> Policy:
    return Policy(PROPRIO_DIM, hist=1, extra_dim=FOOTHOLD_SCAN_DIM).to(device)


class SimRunner:
    def __init__(self, pi: Policy, fpm: FootholdPredictor, mdp: MDP, sim, noise: bool = True):
        self.pi, self.fpm, self.mdp, self.sim, self.noise = pi, fpm, mdp, sim, noise
        dev = sim.device
        self.use_fpm_model = True
        self.xy_drift = torch.zeros(sim.N, 2, device=dev)
        self.z_drift = torch.zeros(sim.N, device=dev)

    @torch.no_grad()
    def observe(self):
        sim, mdp = self.sim, self.mdp
        S = sim.state()
        cmd = sim.command()
        if self.noise:  # new episodes redraw their perception drift (masked: no host sync)
            m = sim.ep_len == 0
            self.xy_drift = torch.where(m.unsqueeze(-1), (2 * torch.rand_like(self.xy_drift) - 1) * XY_DRIFT, self.xy_drift)
            self.z_drift = torch.where(m, (2 * torch.rand_like(self.z_drift) - 1) * Z_DRIFT, self.z_drift)
        o = mdp.proprio(S, sim.a_prev, cmd, sim.xi(), noise=self.noise)
        if self.use_fpm_model:
            fh = self.fpm(mdp.system_state(S, cmd))
        else:
            fh = bootstrap_footholds(sim.N, sim.device)
        K = mdp.context(S)
        return o, perceive(mdp, S, K, fh, self.xy_drift, self.z_drift, self.noise)


class Bundle:
    """Deployed policy and FPM of a training checkpoint (``ckpt_<it>.pt``)."""

    def __init__(self, ck: dict, device: str):
        self.ck, self.args = ck, ck["args"]
        self.pi = make_policy(device)
        self.pi.load_state_dict(ck.get("pi_ema") or ck["pi"])  # the averaged policy
        self.pi.eval()
        self.fpm = FootholdPredictor().to(device)
        self.fpm.load_state_dict(ck["fpm"])
        self.fpm.eval()
        self.device = device


def load_policy_bundle(path: str, device: str) -> Bundle:
    return Bundle(torch.load(path, map_location=device, weights_only=False), device)
