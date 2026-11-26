"""Per-terrain evaluation of a training checkpoint: success and why episodes end, per terrain type and level, with the
training run's terrains, goals and terminations (Isaac Lab terms + the floor / forbidden-contact terminations).

    python scripts/anymal_term_level.py <ckpt> [num_envs] [steps]

Episodes are sampled by the success-guided curriculum over the training terrains. Per (terrain, level): episodes, mean
length (s), success (within --goal_tol of the goal at the episode end), the share of episodes ended by each
termination, mean xy travel and base z change, how often a knee / shank / belly sphere touched the terrain (clearance
< 2 cm, after the first 0.5 s), how often two legs touched each other, and the share of moving steps that went sideways.
"""
import sys
from collections import defaultdict

import torch

from tgi_anymal.env.sim import IsaacLocoSim
from tgi_anymal.policy_io import build_mdp, load_policy_bundle

ck, N, T = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2048, int(sys.argv[3]) if len(sys.argv) > 3 else 2000
dev = "cuda:0"
b = load_policy_bundle(ck, dev)
a = b.args
pk = {k: int(v) for k, v in (x.split(":") for x in a["parkour"].split(",") if x)}
sim = IsaacLocoSim(N, dev, seed=7, stones_share=a["stepping_stones"], pit_step=a["pit_step"], parkour=pk,
                   episode_s=a["episode_s"], goal_dist=tuple(a["goal_dist"]))
run = b.make_runner(sim, noise=True)
meas = build_mdp(sim, dev, roles="knee:allowed,shank:allowed,thigh:tolerated")
names = meas.kin.col_names
part = {p: torch.tensor([p in n for n in names], device=dev) for p in ("knee", "shank", "base")}
mdp_run = build_mdp(sim, dev, roles=a["contact_roles"])
flags = {}


def _extra(S):
    c = mdp_run.min_clearance(S) < a["term_clear"]
    f = S[0][..., 2] < a["term_floor"]
    flags["clear"], flags["floor"] = c, f
    return c | f


sim.extra_termination = _extra
_succ_orig, last_succ = sim._success, {}


def _succ(ids):  # curriculum success, computed inside the reset (before the state is reset); stashed per env
    r = _succ_orig(ids)
    for e, v in zip(ids.tolist(), r.tolist()):
        last_succ[e] = float(v)
    return r


sim._success = _succ
tm = sim.env.termination_manager
tnames = list(tm.active_terms)
gen = sim.cfg.scene.terrain.terrain_generator
ttypes = list(gen.sub_terrains.keys())
z0 = torch.zeros(N, device=dev)
xy0 = torch.zeros(N, 2, device=dev)
cnt = {p: torch.zeros(N, device=dev) for p in part}
selft = torch.zeros(N, device=dev)  # steps with two legs touching (sphere / foot gap < 0)
side = torch.zeros(N, device=dev)  # moving steps with the velocity > 45 deg off the body axis
mov = torch.zeros(N, device=dev)
new = torch.ones(N, dtype=torch.bool, device=dev)
age = torch.zeros(N, device=dev)
acc = defaultdict(lambda: defaultdict(float))
for k in range(T):
    S = sim.state()
    if new.any():
        z0 = torch.where(new, S[0][:, 2], z0)
        xy0 = torch.where(new.unsqueeze(-1), S[0][:, :2], xy0)
        for p in part:
            cnt[p] = torch.where(new, torch.zeros_like(cnt[p]), cnt[p])
        age = torch.where(new, torch.zeros_like(age), age)
        selft, side, mov = (torch.where(new, torch.zeros_like(x_), x_) for x_ in (selft, side, mov))
    age += 1
    with torch.no_grad():
        Km = meas.context(S)
        clr = Km["clr"]
        selft += ((Km["self_gap"].amin(-1) < 0.0) & (age > 25)).float()
        vb = (S[1].transpose(-1, -2) @ S[2].unsqueeze(-1)).squeeze(-1)
        sp = vb[:, :2].norm(dim=-1)
        m_ = (sp > 0.2) & (age > 25)
        mov += m_.float()
        side += (m_ & (vb[:, 1].abs() > vb[:, 0].abs())).float()
    for p, m in part.items():
        cnt[p] += ((clr[:, m].amin(-1) < 0.02) & (age > 25)).float()  # skip the 0.5 s spawn settling
    S_prev = S
    out = sim.step(run.act())
    new = sim.ep_len == 0
    if "ep_ids" not in out:
        continue
    ids = out["ep_ids"]
    fired = {n: tm.get_term(n)[ids].clone() for n in tnames}
    for n in ("clear", "floor"):
        fired["x_" + n] = flags[n][ids].clone()
    p_end = S_prev[0][ids]
    dz = (p_end[:, 2] - z0[ids]).tolist()
    dxy = (p_end[:, :2] - xy0[ids]).norm(dim=-1).tolist()
    succ = [last_succ.pop(e, float("nan")) for e in ids.tolist()]
    lv, ty, ln = out["ep_level"].tolist(), out["ep_type"].tolist(), out["ep_len"].tolist()
    for i, e in enumerate(ids.tolist()):
        r = acc[(ttypes[ty[i]], str(lv[i]))]
        r["n"] += 1
        r["len"] += ln[i] * sim.dt
        r["succ"] += float(succ[i])
        r["dxy"] += dxy[i]
        r["dz"] += dz[i]
        for n, v in fired.items():
            r["T:" + n] += float(v[i])
        for p in part:
            r["c:" + p] += float(cnt[p][e] > 0)
        r["selfcol"] += float(selft[e] > 0)
        r["side"] += float(side[e] / mov[e].clamp(min=1))
print(f"run args: parkour {a['parkour']} stepping_stones {a['stepping_stones']}")
keys = sorted({k for r in acc.values() for k in r if k.startswith("T:")})
print(f"{'terrain':>22} {'lvl':>4} {'n':>4} {'len s':>5} {'succ':>5} {'dxy':>5} {'dz':>6} " + " ".join(f"{k[2:]:>10}" for k in keys)
      + f" {'knee':>5} {'shank':>5} {'belly':>5} {'self':>5} {'side':>5}")
for (t, band), r in sorted(acc.items()):
    n = r["n"]
    print(f"{t:>22} {band:>4} {int(n):>4} {r['len'] / n:5.1f} {r['succ'] / n:5.2f} {r['dxy'] / n:5.2f} {r['dz'] / n:6.2f} "
          + " ".join(f"{r[k] / n:10.2f}" for k in keys)
          + f" {r['c:knee'] / n:5.2f} {r['c:shank'] / n:5.2f} {r['c:base'] / n:5.2f} {r['selfcol'] / n:5.2f} {r['side'] / n:5.2f}")
print("EVAL_OK")
