"""Terrain Grounded Imagination training loop.

Each iteration collects simulator steps, fits the contact-force world model (3D terrain features from the heightmap and
the terrain distance field at every step) and the FPM, then backpropagates the return of ``horizon`` imagined steps
through the world model into the policy, with a critic bootstrap and the gradient cut at imagined touchdowns on
unsupported terrain.
"""

from __future__ import annotations

import copy
import argparse
import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn

from ..constants import CONTACT_BODIES, NUM_J
from ..env.sim import IsaacLocoSim
from ..foot_pred import FootholdPredictor, FPMTrainer
from ..mdp import RewardCfg, normalize_xi, unflatten_state
from ..mdp.interface import XY_DRIFT, Z_DRIFT, mirror_joints, mirror_proprio, mirror_scan
from ..policy_io import bootstrap_footholds, build_mdp, make_policy, perceive
from ..world_model import Critic, ImpulseWorldModel, ReplayBuffer, networks
from .collector import Collector
from .logger import TrainLogger
from .optim import cosine_lr, load_optimizer_state, make_optimizer, set_lr, skip_if_nonfinite

# floor of the per-dimension error scale: position, rotation, velocity, angular velocity, q, qd
ERR_FLOOR = (3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3 + (3e-3,) * NUM_J + (5e-2,) * NUM_J


class Profiler:
    """Per-phase wall time in ms per iteration (``TGI_PROF=1``); syncs only the update stream."""

    def __init__(self, stream: torch.cuda.Stream | None):
        self.enabled = os.environ.get("TGI_PROF") == "1"
        self.stream, self.acc, self.t = stream, {}, time.perf_counter()

    def tick(self, name: str):
        if not self.enabled:
            return
        self.stream.synchronize() if self.stream is not None else torch.cuda.synchronize()
        t = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + t - self.t
        self.t = t

    def report(self, iters: int):
        if self.enabled:
            total = sum(self.acc.values())
            print("   prof ms/it: " + " ".join(f"{k}={1e3 * v / iters:.1f}" for k, v in self.acc.items())
                  + f" total={1e3 * total / iters:.1f}", flush=True)
            self.acc.clear()


def requires_grad(*modules: nn.Module | None, flag: bool):
    for m in modules:
        if m is not None:
            m.requires_grad_(flag)


class Trainer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.dev = dev = "cuda:0"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        if cfg.amp == "bf16" and torch.cuda.get_device_capability()[0] < 8:
            print("no native bf16 on this GPU (compiled bf16 graphs would fall back to eager): --amp none", flush=True)
            cfg.amp = "none"
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        os.makedirs(cfg.log_dir, exist_ok=True)
        json.dump(vars(cfg), open(os.path.join(cfg.log_dir, "args.json"), "w"), indent=2)

        # simulator first; torch gets the rest of --vram_gb
        free0, total = torch.cuda.mem_get_info()
        self.sim = sim = IsaacLocoSim(cfg.num_envs, dev, seed=cfg.seed, stones_share=cfg.stepping_stones,
                                      sgs_spawns=cfg.sgs_spawns, sgs_dirs=cfg.sgs_dirs, sgs_window=cfg.sgs_window,
                                      sgs_neighbor=cfg.sgs_neighbor, sgs_target=cfg.sgs_target, sgs_give=cfg.sgs_give,
                                      pit_step=cfg.pit_step,
                                      parkour={k: int(v) for k, v in (x.split(":") for x in cfg.parkour.split(",") if x)},
                                      episode_s=cfg.episode_s, goal_dist=tuple(cfg.goal_dist), goal_tol=cfg.goal_tol)
        sim_gb = (free0 - torch.cuda.mem_get_info()[0]) / 1e9
        torch_gb = cfg.vram_gb - sim_gb
        assert torch_gb > 1.0, f"--vram_gb {cfg.vram_gb} is too small: the simulator alone uses {sim_gb:.2f} GB"
        torch.cuda.set_per_process_memory_fraction(min(torch_gb * 1e9 / total, 1.0))
        print(f"VRAM budget {cfg.vram_gb:.1f} GB: simulator {sim_gb:.2f} GB, torch cap {torch_gb:.2f} GB", flush=True)

        reward_cfg = RewardCfg(term_floor=cfg.term_floor, term_angle=cfg.term_angle, term_height=cfg.term_height,
                               term_clear=cfg.term_clear, w_term=cfg.w_term, w_self=cfg.w_self,
                               self_margin=cfg.self_margin, w_face=cfg.w_face, w_lat=cfg.w_lat, w_impact=cfg.w_impact,
                               impact_n=cfg.impact_n, w_power=cfg.w_power)
        self.mdp = mdp = build_mdp(sim, dev, reward_cfg, field_window=cfg.field_window, field_hole=cfg.field_hole,
                                   hm_res=cfg.hm_res, roles=cfg.contact_roles)
        print(f"3D contact world model: {len(mdp.terrain.col_roles)} collision spheres "
              f"{dict(Counter(mdp.terrain.col_roles))}, terrain input {mdp.terrain.wm_dim}", flush=True)

        def _extra(S):  # the same floor / forbidden-contact terminations in the simulator as in imagination
            return (S[0][..., 2] < cfg.term_floor) | (mdp.min_clearance(S) < cfg.term_clear)

        sim.extra_termination = _extra

        self.wm = ImpulseWorldModel(mdp.terrain.wm_dim, tuple(cfg.wm_hidden), mdp.kin.joint_mask(CONTACT_BODIES)).to(dev)
        print(f"contact-force world model: {len(CONTACT_BODIES)} contact bodies {CONTACT_BODIES}; joint mask\n"
              f"{self.wm.jmask.int().tolist()}", flush=True)
        self.pi = make_policy(dev)  # Isaac joint order
        self.critic = Critic(mdp.critic_dim).to(dev)
        self.fpm = FootholdPredictor().to(dev)
        self.fpm_trainer = FPMTrainer(self.fpm)
        self.opt_wm = make_optimizer("muon", self.wm, cfg.lr, muon_lr_mult=cfg.muon_lr_mult,
                                     muon_min_dim=cfg.muon_min_dim)
        self.pi_mult0 = cfg.pi_muon_mult
        self.opt_pi = make_optimizer("muon", self.pi, cfg.pi_lr, betas=(0.9, 0.95), muon_lr_mult=self.pi_mult0)
        # the deployed policy: an average of the trained one's iterates (not trained itself)
        self.pi_ema = copy.deepcopy(self.pi).requires_grad_(False)
        self.opt_v = make_optimizer("adam", self.critic, cfg.lr)
        for name, opt in (("world model", self.opt_wm), ("policy", self.opt_pi)):
            print(f"{name} optimizer: {opt.describe()}", flush=True)

        self.start_it, self.wall_offset, sim_steps = 0, 0.0, 0
        if cfg.resume:
            self.start_it, self.wall_offset, sim_steps = self._load(cfg.resume)
        self.t_start = time.time() - self.wall_offset
        self.err_floor = torch.tensor(ERR_FLOOR, device=dev)
        self._compile()

        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, dev)
        self.last_force = torch.zeros(6, device=dev)
        names = list(sim.cfg.scene.terrain.terrain_generator.sub_terrains.keys())
        self.logger = TrainLogger(cfg.log_dir, resume=self.start_it > 0, log_every=cfg.log_every, terrain_names=names,
                                  device=dev)
        self.collector = Collector(sim, mdp, self.pi, self.fpm, self.wm, self.buffer, self.logger, cfg.action_noise,
                                   threaded=bool(cfg.async_collect))
        self.collector.steps = sim_steps
        self.prof = Profiler(self.collector.train_stream)
        # overlap only once the CUDA graphs are recorded
        self.overlap_from = max(cfg.async_after, cfg.wm_only_iters + 50, cfg.fpm_bootstrap + 50)
        if cfg.async_collect:
            print(f"async collection: the simulator overlaps the updates from iteration "
                  f"{max(self.overlap_from, self.start_it)}", flush=True)
        self.last_sym_loss = torch.zeros((), device=dev)
        self.last_event_share = torch.zeros((), device=dev)

    def _load(self, path: str) -> tuple[int, float, int]:
        ck = torch.load(path, map_location=self.dev)
        self.wm.load_state_dict(ck["wm"])
        self.pi.load_state_dict(ck["pi"])
        self.critic.load_state_dict(ck["critic"])
        self.fpm.load_state_dict(ck["fpm"])
        for key, opt in (("opt_wm", self.opt_wm), ("opt_pi", self.opt_pi), ("opt_v", self.opt_v)):
            load_optimizer_state(opt, ck[key], self.dev)
        self.pi_ema.load_state_dict(ck["pi_ema"])
        return ck["iteration"], ck["wall_time"], ck["sim_steps"]

    def save(self, it: int):
        for b_ema, b in zip(self.pi_ema.buffers(), self.pi.buffers()):  # observation norms: data statistics
            b_ema.copy_(b)
        ck = {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "pi_ema": self.pi_ema.state_dict(),
              "critic": self.critic.state_dict(), "fpm": self.fpm.state_dict(), "fpm_active": self.fpm_active(it),
              "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
              "iteration": it, "wall_time": time.time() - self.t_start, "sim_steps": self.collector.steps,
              "args": vars(self.cfg), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(self.cfg.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        keep = ("pi", "wm", "critic", "fpm", "fpm_active", "iteration", "wall_time", "sim_steps", "args", "saved_at",
                "pi_ema")  # the policy checkpoint, without the optimizer states
        torch.save({k: ck[k] for k in keep}, os.path.join(self.cfg.log_dir, f"ckpt_{it:06d}.pt"))

    def fpm_active(self, it: int) -> bool:
        return it >= self.cfg.fpm_bootstrap

    def _compile(self):
        """The per-step functions, as closures (torch.compile; CUDA graphs for the world-model loss and imagination)."""
        cfg, mdp, wm, pi, fpm, dev, dt = self.cfg, self.mdp, self.wm, self.pi, self.fpm, self.dev, self.sim.dt
        err_floor, jmap = self.err_floor, mdp.jmap

        def wm_loss(s, a, a_prev, xi, s1, c1, mask, f1):
            S = unflatten_state(s[:, 0])
            preds, logits, forces, loads = [], [], [], []
            for j in range(cfg.wm_window):
                tf, geo = mdp.wm_inputs(S)
                S, lg, F, P = wm.step_full(S, tf, mdp.joint_target(a[:, j]), mdp.joint_target(a_prev[:, j]),
                                           normalize_xi(xi[:, j]), geo)
                forces.append(F), loads.append(P)
                preds.append(S), logits.append(lg)
            S_hat = tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))
            logits = torch.stack(logits, 1)
            std = torch.maximum(wm.out_norm.std(), err_floor)
            e = ((wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2).mean(-1)
            B = s1.shape[0]
            l_dyn = (e * mask).sum() / B
            acc = ((((logits > 0).float() == c1).float().mean(-1)) * mask).sum() / mask.sum()
            # contact forces and loaded probabilities of every contact body (the feet's are the foot contacts)
            F, P = torch.stack(forces, 1), torch.stack(loads, 1)  # (B, k, nb, 3), (B, k, nb)
            F0 = wm.F0
            l_f = ((F / F0 - f1 / F0) ** 2).sum(-1).mean(-1)
            loaded = (f1.norm(dim=-1) > 1.0).float()
            bce_b = nn.functional.binary_cross_entropy(P.clamp(1e-5, 1 - 1e-5), loaded, reduction="none").mean(-1)
            l_f, l_cb = (l_f * mask).sum() / B, (bce_b * mask).sum() / B
            with torch.no_grad():  # non-foot bodies: accuracy and recall of "loaded"; force error (N) of all bodies
                m3 = mask.unsqueeze(-1)
                nf_pred, nf_lab = (P[..., 4:] > 0.5).float(), loaded[..., 4:]
                nf_acc = ((nf_pred == nf_lab).float() * m3).sum() / (m3.sum() * nf_lab.shape[-1])
                nf_rec = (nf_pred * nf_lab * m3).sum() / (nf_lab * m3).sum().clamp(min=1)
                f_err = ((F - f1).norm(dim=-1).mean(-1) * mask).sum() / mask.sum()
                nf_share = (nf_lab * m3).sum() / (m3.sum() * nf_lab.shape[-1])
                # foot impact term (per foot-step, units of 100 N) on the simulator's and the model's forces
                imp_s = (torch.relu(f1[..., :4, :].norm(dim=-1) - cfg.impact_n) / 100.0).mean(-1)
                imp_w = (torch.relu(F[..., :4, :].norm(dim=-1) - cfg.impact_n) / 100.0).mean(-1)
                imp_s, imp_w = (imp_s * mask).sum() / mask.sum(), (imp_w * mask).sum() / mask.sum()
            stats = torch.stack((f_err, nf_acc, nf_rec, nf_share, imp_s, imp_w))
            return l_dyn + cfg.contact_w * l_cb + cfg.force_w * l_f, l_dyn, e[:, -1], acc, stats

        def imag_step(S, K, o, extra, a_prev, cmd, xi, xi_n, xy_d, z_d, use_fpm: bool, eps):
            mu = pi(o.unsqueeze(1), extra)  # Isaac joint order
            a_i = mu + eps  # reparameterized exploration noise (zero without the likelihood-ratio term)
            a = jmap.to_mj(a_i)
            S1, lg, F1, _ = wm.step_full(S, K["tf"], mdp.joint_target(a), mdp.joint_target(a_prev), xi_n, K["geo"])
            K1 = mdp.context(S1)
            # foot forces of this transition (contact bodies start with the feet)
            K1 = dict(K1, ff=torch.sqrt((F1[..., :4, :] ** 2).sum(-1) + 1e-6))
            o1 = mdp.proprio(S1, a, cmd, xi)
            with torch.no_grad():
                fh = fpm(mdp.system_state(S1, cmd).detach()) if use_fpm else bootstrap_footholds(o.shape[0], dev)
            extra = perceive(mdp, S1, K1, fh, xy_d, z_d)
            logp = -((a_i.detach() - mu) ** 2).sum(-1) / (2 * cfg.imag_noise**2)
            return S1, K1, torch.sigmoid(lg), a, o1, extra, logp

        def reward(S, K, c, a, a_prev, cmd, prev):
            d = mdp.soft_termination(S, K)
            return mdp.reward(S, K, c, a, a_prev, cmd, dt, prev=prev, d=d), d

        def critic_input(S, c, cmd, xi_n):
            return mdp.critic_features(S, mdp.context(S), c, cmd, xi_n)

        def compiled(f, graphs: bool):
            if not cfg.compile:
                return f
            return torch.compile(f, mode="reduce-overhead") if graphs else torch.compile(f)

        self.wm_graphs = bool(cfg.compile and cfg.cudagraphs)
        self.imag_graphs = bool(self.wm_graphs and cfg.cg_imag)
        self.wm_loss = compiled(wm_loss, self.wm_graphs)
        self.imag_step = compiled(imag_step, self.imag_graphs)
        self.reward = compiled(reward, False)
        self.critic_input = compiled(critic_input, False)

    def wm_update(self):
        cfg = self.cfg
        keys = ("s", "s1", "a", "a_prev", "xi", "c1", "done", "f1")
        w = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window, keys=keys)
        self.prof.tick("wm_sample")
        if self.wm_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        loss, l_dyn, e_last, acc, fstats = self.wm_loss(w["s"], w["a"], w["a_prev"], w["xi"], w["s1"], w["c1"], w["mask"],
                                                        w["f1"])
        self.last_force = fstats.detach().clone()
        self.prof.tick("wm_fwd")
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        self.prof.tick("wm_bwd")
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        self.prof.tick("wm_opt")
        m = w["mask"][:, -1]
        # new tensors: CUDA-graph outputs are overwritten by the next replay
        return l_dyn.detach() / cfg.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), acc.detach().clone()

    def policy_update(self, use_fpm: bool):
        cfg, mdp, dev, pi, critic = self.cfg, self.mdp, self.dev, self.pi, self.critic
        B, H, gamma = cfg.pi_batch, cfg.horizon, cfg.gamma
        if self.imag_graphs:
            torch.compiler.cudagraph_mark_step_begin()

        # start from replayed states, with fresh perception drift
        s, h = self.buffer.sample_starts(B, 2)
        S = unflatten_state(s["s"])
        xi, cmd, a_prev = s["xi"], s["cmd"], s["a_prev"]
        # goal command per imagined step: the remaining time counts down
        cmds = [torch.cat((cmd[:, :3], (cmd[:, 3:] - k * self.sim.dt).clamp(min=0.0)), -1) for k in range(H + 1)]
        xi_n = normalize_xi(xi)
        xy_d = (2 * torch.rand(B, 2, device=dev) - 1) * XY_DRIFT
        z_d = (2 * torch.rand(B, device=dev) - 1) * Z_DRIFT
        K = mdp.context(S)
        with torch.no_grad():
            o = mdp.proprio(S, a_prev, cmd, xi)
            c0 = h["c1"][:, 0]
            fh = self.fpm(mdp.system_state(S, cmd)) if use_fpm else bootstrap_footholds(B, dev)
            extra = perceive(mdp, S, K, fh, xy_d, z_d)
        requires_grad(self.wm, critic, self.fpm, flag=False)
        S0, K0 = S, K
        Ss, Ks, Cs, As, Aps, Lps = [], [], [], [], [], []
        Os, Es = [o.detach()], [extra.detach()]
        use_lr = cfg.lr_coef > 0
        self.prof.tick("pi_setup")
        for t in range(H):
            Aps.append(a_prev)
            eps = cfg.imag_noise * torch.randn(B, NUM_J, device=dev) if use_lr else torch.zeros(B, NUM_J, device=dev)
            S, K, c, a_prev, o, extra, logp = self.imag_step(S, K, o, extra, a_prev, cmds[t + 1], xi, xi_n, xy_d, z_d,
                                                             use_fpm, eps)
            Ss.append(S), Ks.append(K), Cs.append(c), As.append(a_prev), Lps.append(logp)
            if t < H - 1:
                Os.append(o.detach()), Es.append(extra.detach())
        self.prof.tick("pi_imag_fwd")

        # rewards and soft terminations of all imagined steps at once
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        K_all = {k: flat([Kt[k] for Kt in Ks]) for k in Ks[0] if k != "geo"}  # (geo: the world model's input only)
        cmd_all = flat(cmds[1:])
        prev = (flat([S0[2]] + [St[2] for St in Ss[:-1]]), flat([S0[3]] + [St[3] for St in Ss[:-1]]),
                flat([S0[5]] + [St[5] for St in Ss[:-1]]), flat([K0["fv"]] + [Kt["fv"] for Kt in Ks[:-1]]))  # accelerations
        r, d = self.reward(S_all, K_all, flat(Cs), flat(As), flat(Aps), cmd_all, prev)
        r = r * (cmd_all[:, 3] > 0).float()  # past the deadline the episode is over (time out): no reward
        r, d = r.view(H, B), d.view(H, B)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        surv = alive[-1]
        r_mean = (c_t * r).detach().sum(0) / H / self.sim.dt
        rews, dones = r.detach(), d.detach()

        # policy objective, differentiable through the world model:
        # G_t = r_t + gamma (1 - d_t) ((1 - l_t) V(s_t+1) + l_t G_t+1); l_t = 1 except at cut touchdowns (0)
        xi_all = xi_n.unsqueeze(0).expand(H, -1, -1).flatten(0, 1)
        Vg = critic(self.critic_input(S_all, flat(Cs), cmd_all, xi_all)).view(H, B)
        with torch.no_grad():  # cut the gradient through the model where it is least reliable
            Ct = torch.stack(Cs)
            touchdown = Ct * (1 - torch.cat((c0.unsqueeze(0), Ct[:-1]), 0))  # soft, (H, B, n_feet)
            unsupported = torch.stack([Kt["unsup"] for Kt in Ks])[..., 0]  # unsupported footprint share
            event = (touchdown * unsupported).amax(-1) > cfg.pg_event
            lam = torch.where(event, torch.zeros((H, B), device=dev), torch.ones((H, B), device=dev))
        self.last_event_share = event.float().mean()
        G = r[-1] + gamma * (1 - d[-1]) * Vg[-1]
        for t in range(H - 2, -1, -1):
            G = r[t] + gamma * (1 - d[t]) * ((1 - lam[t]) * Vg[t] + lam[t] * G)
        J = G

        # critic values and lambda-return targets of the imagined transitions
        with torch.no_grad():
            stack0 = lambda x0, xs: torch.cat((x0.unsqueeze(0), torch.stack(xs)), 0).flatten(0, 1)  # noqa: E731
            S_c = tuple(stack0(S0[i], [St[i].detach() for St in Ss]) for i in range(6))
            c_c = stack0(c0, [c.detach() for c in Cs])
            cmd_c = torch.stack(cmds).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(H + 1, -1, -1).flatten(0, 1)
            X = self.critic_input(S_c, c_c, cmd_c, xi_c).view(H + 1, B, -1)
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(H - 1, -1, -1):
                R = rews[t] + gamma * (1 - dones[t]) * ((1 - cfg.lam) * V[t + 1] + cfg.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])

        loss = -J.mean()
        if use_lr:  # likelihood-ratio term: credit actions by their lambda-return advantage
            adv = targets - V[:-1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            loss = loss - cfg.lr_coef * (c_t.detach() * torch.stack(Lps) * adv).mean()
        if cfg.sym_coef > 0:  # mirror consistency: pi(mirror(o)) = mirror(pi(o))
            Of, Ef = torch.cat(Os), torch.cat(Es)
            idx = torch.randint(0, Of.shape[0], (min(4096, Of.shape[0]),), device=dev)
            Of, Ef = Of[idx], Ef[idx]
            a_o = pi(Of.unsqueeze(1), Ef)
            a_m = pi(mirror_proprio(Of).unsqueeze(1), mirror_scan(Ef))
            sym_loss = ((a_m - mirror_joints(a_o)) ** 2).mean()
            loss = loss + cfg.sym_coef * sym_loss
            self.last_sym_loss = sym_loss.detach()
        self.prof.tick("pi_return_sym_fwd")
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        skip_if_nonfinite(self.opt_pi, gn)
        self.opt_pi.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(self.pi_ema.parameters()), list(pi.parameters()), 1.0 - cfg.pi_ema)
        requires_grad(self.wm, critic, self.fpm, flag=True)
        self.prof.tick("pi_backward_step")

        critic.norm.update(X)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // cfg.critic_minibatches
        v_loss = torch.zeros((), device=dev)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                self.opt_v.step()
        self.prof.tick("critic")
        return loss.detach(), v_loss.detach(), r_mean.mean(), gn.detach(), surv.mean().detach()

    def run(self):
        cfg, collector, logger = self.cfg, self.collector, self.logger
        collector.collect(max(cfg.warmup_steps, cfg.wm_window + 1), self.fpm_active(self.start_it))
        self.t_start = time.time() - self.wall_offset  # training time, without the setup
        t_job = logger.t_log = time.time()
        step_acc = 0.0
        for it in range(self.start_it, cfg.iterations):
            set_lr(self.opt_wm, cosine_lr(cfg.lr, it, cfg.lr_decay_iters))
            if cfg.pi_mult_decay > 0:  # large policy Muon steps first, then x1
                u = min(1.0, (it - self.start_it) / cfg.pi_mult_decay)
                self.opt_pi.muon_lr_mult = self.pi_mult0 + (1.0 - self.pi_mult0) * u
            set_lr(self.opt_pi, cosine_lr(cfg.pi_lr, it, cfg.lr_decay_iters))
            set_lr(self.opt_v, cosine_lr(cfg.lr, it, cfg.lr_decay_iters))
            step_acc += 1.0 / cfg.utd
            n_steps = int(step_acc)
            step_acc -= n_steps
            self.prof.tick("other")

            # in threaded mode the new transitions reach the buffer after this iteration's updates
            job = collector.start(n_steps, self.fpm_active(it), overlap=collector.threaded and it >= self.overlap_from)
            if not collector.threaded:
                collector.finish(job)
            self.prof.tick("collect")
            for _ in range(cfg.wm_batches):
                wm_stats = self.wm_update()
            logger.add_wm(*wm_stats)  # the last update of the iteration
            self.prof.tick("wm")
            if self.buffer.size > 100:
                if self.fpm_trainer.labels is None or it % cfg.fpm_label_every == 0:
                    self.fpm_trainer.refresh_labels(self.buffer, self.mdp.kin)
                self.prof.tick("fpm_labels")
                logger.add_fpm(self.fpm_trainer.update(self.buffer, self.mdp, cfg.fpm_batch, steps=cfg.fpm_steps))
            self.prof.tick("fpm")
            if it >= cfg.wm_only_iters:
                logger.add_policy(*self.policy_update(self.fpm_active(it)))
            if collector.threaded:
                collector.finish(job)
                self.prof.tick("collect_wait")

            if (it + 1) % cfg.log_every == 0:
                extra = {}
                if cfg.sym_coef > 0:
                    extra["sym_loss"] = float(self.last_sym_loss)
                extra["pg_event steps"] = float(self.last_event_share)
                f = self.last_force.tolist()
                extra.update({"f_err N": f[0], "nonfoot acc": f[1], "nonfoot recall": f[2], "nonfoot loaded": f[3],
                              "impact sim": f[4], "impact wm": f[5]})
                logger.write(it + 1, (time.time() - self.t_start) / 3600, collector.steps,
                             cosine_lr(cfg.pi_lr, it, cfg.lr_decay_iters), extra, sim=self.sim)
                self.prof.report(cfg.log_every)
            if (it + 1) % cfg.ckpt_every == 0 or it + 1 == cfg.iterations:
                self.save(it + 1)
            if (time.time() - t_job) / 3600 > cfg.max_hours:
                self.save(it + 1)
                break
