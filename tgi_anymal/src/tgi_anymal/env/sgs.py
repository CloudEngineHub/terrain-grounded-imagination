"""Success-guided sampling (SGS) terrain curriculum (Zhang et al., arXiv 2610.12465).

Each configuration (terrain cell x spawn patch x travel direction) keeps its last outcomes, p_i = mean(window), and
is sampled with

    w_i = max((p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)), eps),    P(i) = softmax(log(w_i + eps) / T)

within each terrain type (``group``; the paper's Eq. 2 is one softmax over all configurations), and the terrain
types get their shares ``group_prior``; w_i is floored at eps and T_eff = max(T, 1) as in the paper's App. B.2.
Unvisited configurations count as p = 0. Not in the paper: ``neighbor`` lower-bounds a configuration's success by that
of the same configuration one level easier, and ``give`` > 0 reattributes type shares by need: a type gives away up to
``give`` of its share as it gets solved (solvedness s_g = mean effective success over its configurations, mapped
linearly from ``solved`` = (lo, hi) to 0..1), and the freed share goes to the types in proportion to
prior_g * (1 - s_g). Success (sim.py): within --goal_tol of the goal at the episode end.
Paper's locomotion setting: H = 100, t = 0.66, kappa = 5, T = 2, eps = 1e-8.
"""

from __future__ import annotations

import torch


class SuccessGuidedSampler:
    def __init__(self, num_configs: int, device, group: torch.Tensor, group_prior: torch.Tensor, window: int = 100,
                 target: float = 0.66, kappa: float = 5.0, temperature: float = 2.0, eps: float = 1e-8,
                 prev: torch.Tensor | None = None, neighbor: float = 0.0, give: float = 0.0,
                 solved: tuple[float, float] = (0.6, 0.95)):
        """group: terrain type of each configuration; group_prior: share per type before reattribution; prev: the same
        configuration one level easier (-1 at the easiest), whose success lower-bounds this one's (times ``neighbor``).
        """
        self.n, self.H = num_configs, window
        self.group, self.group_prior = group, group_prior
        self.t, self.kappa, self.T, self.eps = target, kappa, max(temperature, 1.0), eps
        self.outcomes = torch.zeros(num_configs, window, device=device)
        self.count = torch.zeros(num_configs, dtype=torch.long, device=device)  # total outcomes recorded
        self.prev, self.neighbor = prev, neighbor
        self.give, self.solved = give, solved
        self.levels = 0
        if prev is not None:  # number of levels = longest prev chain
            i = torch.arange(num_configs, device=device)
            while bool((i >= 0).any()):
                i = torch.where(i >= 0, prev[i.clamp(min=0)], i)
                self.levels += 1

    def success_rate(self) -> torch.Tensor:
        filled = self.count.clamp(max=self.H)
        return self.outcomes.sum(-1) / filled.clamp(min=1)  # unvisited: 0

    def effective_success(self) -> torch.Tensor:
        p = self.success_rate()
        if self.prev is None or self.neighbor <= 0:
            return p
        for _ in range(self.levels):  # p_eff[l] = max(p[l], neighbor * p_eff[l - 1])
            p = torch.maximum(p, self.neighbor * torch.where(self.prev >= 0, p[self.prev.clamp(min=0)], torch.zeros_like(p)))
        return p

    def group_shares(self, p: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(type shares after need-based reattribution, solvedness per type); the fixed prior if give == 0."""
        if p is None:
            p = self.effective_success()
        G = len(self.group_prior)
        cnt = torch.zeros(G, device=p.device).index_add_(0, self.group, torch.ones_like(p))
        s = torch.zeros(G, device=p.device).index_add_(0, self.group, p) / cnt.clamp(min=1)
        if self.give <= 0:
            return self.group_prior, s
        lo, hi = self.solved
        f = self.give * ((s - lo) / (hi - lo)).clamp(0.0, 1.0)  # share fraction each type gives away
        pool = (self.group_prior * f).sum()
        need = self.group_prior * (1.0 - s)
        share = self.group_prior * (1.0 - f) + pool * need / need.sum().clamp(min=1e-12)
        return share / share.sum(), s

    def probs(self) -> torch.Tensor:
        p = self.effective_success()
        w = (p + self.eps) ** (self.kappa * self.t) * (1 - p + self.eps) ** (self.kappa * (1 - self.t))
        logit = torch.log(w.clamp(min=self.eps) + self.eps) / self.T
        e = torch.exp(logit - logit.max())
        z = torch.zeros_like(self.group_prior).index_add_(0, self.group, e)
        share, _ = self.group_shares(p)
        return share[self.group] * e / z[self.group]

    def record(self, cfg_ids: torch.Tensor, success: torch.Tensor):
        # circular buffers; several outcomes for the same config in one call go to consecutive slots
        ids, order = torch.sort(cfg_ids, stable=True)
        pos = torch.arange(len(ids), device=ids.device)
        first = torch.full((self.n,), len(ids), device=ids.device, dtype=torch.long).scatter_reduce(0, ids, pos, "amin")
        slot = (self.count[ids] + pos - first[ids]) % self.H
        self.outcomes[ids, slot] = success[order].float()
        self.count.index_add_(0, ids, torch.ones_like(ids))

    def sample(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.probs(), n, replacement=True)

    def stats(self) -> dict[str, float]:
        P, p = self.probs(), self.success_rate()
        visited = self.count > 0
        return {"eff_configs": float(torch.exp(-(P * P.clamp(min=1e-30).log()).sum())),
                "mean_success": float(p[visited].mean()) if visited.any() else 0.0,
                "visited": float(visited.float().mean()),
                "p_target_band": float(P[(p > 0.3) & (p < 0.9)].sum())}
