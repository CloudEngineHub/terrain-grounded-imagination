"""3D distance field of the terrain, queried at arbitrary 3D points (signed distance and surface normal).

The terrain is a grid of column tops (the heightmap, ray-cast from above); each cell is an exact box from its top down,
so stone edges and the flanks of a block or a wall are sharp at the grid resolution, and a point beside a wall is as
far from it as it really is, not as far as the ground below it.

The query is differentiable in the point. The nearest column is selected without gradient over a small window, then
the distance to that one column is recomputed with gradient, so the graph holds one cell per point, not the window.
"""

from __future__ import annotations

import torch


class GeometryField:
    def __init__(self, tops: torch.Tensor, origin_xy: tuple[float, float], res: float, window: int = 3, clip: float = 0.6):
        """tops: (NX, NY) column top heights at nodes origin + (i, j) * res; window: cells searched each way."""
        self.h = tops
        self.nx, self.ny = tops.shape
        self.ox, self.oy = origin_xy
        self.res, self.clip = res, clip
        k = torch.arange(-window, window + 1, device=tops.device)
        di, dj = torch.meshgrid(k, k, indexing="ij")
        self.di, self.dj = di.flatten(), dj.flatten()

    def _column(self, x, i, j):
        """Signed distance and unit normal from points x (..., 3) to the columns (i, j) (same leading shape)."""
        half = 0.5 * self.res
        ex = x[..., 0] - (self.ox + i * self.res)
        ey = x[..., 1] - (self.oy + j * self.res)
        dx, dy = torch.relu(ex.abs() - half), torch.relu(ey.abs() - half)
        dz = x[..., 2] - self.h[i.clamp(0, self.nx - 1), j.clamp(0, self.ny - 1)]
        lat2 = dx * dx + dy * dy
        above = dz > 0
        beside = ~above & (lat2 > 0)
        d_out = torch.sqrt(lat2 + torch.relu(dz) ** 2 + 1e-12)
        d = torch.where(above, d_out, torch.where(beside, torch.sqrt(lat2 + 1e-12), dz))
        return d, (ex, ey, dx, dy, dz, above, beside)

    def query(self, x: torch.Tensor, normal: bool = False):
        """Signed distance (..., ) to the nearest surface, clipped to ``clip``; with ``normal`` also its unit
        normal (..., 3) (no gradient through the normal). Differentiable w.r.t. x."""
        xd = x.detach()
        i0 = ((xd[..., 0] - self.ox) / self.res).round().long()
        j0 = ((xd[..., 1] - self.oy) / self.res).round().long()
        with torch.no_grad():  # which column is nearest
            I, J = i0.unsqueeze(-1) + self.di, j0.unsqueeze(-1) + self.dj
            d_all, _ = self._column(xd.unsqueeze(-2), I, J)
            best = d_all.argmin(-1, keepdim=True)
            i, j = I.gather(-1, best).squeeze(-1), J.gather(-1, best).squeeze(-1)
        d, (ex, ey, dx, dy, dz, above, beside) = self._column(x, i, j)
        n = None
        if normal:
            with torch.no_grad():
                sx, sy = torch.sign(ex) * dx, torch.sign(ey) * dy
                up = torch.stack((torch.zeros_like(d), torch.zeros_like(d), torch.ones_like(d)), -1)
                n_out = torch.stack((sx, sy, torch.relu(dz)), -1)
                n_side = torch.stack((sx, sy, torch.zeros_like(d)), -1)
                n = torch.where(above.unsqueeze(-1), n_out, torch.where(beside.unsqueeze(-1), n_side, up))
                n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        d = d.clamp(max=self.clip)
        return (d, n) if normal else d
