"""Global terrain heightmap: raycast once on the GPU, then queried (bilinear, differentiable) anywhere."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp


class HeightMap:
    def __init__(self, heights: torch.Tensor, origin_xy: tuple[float, float], res: float):
        self.h = heights  # (NX, NY), world z
        self.ox, self.oy = origin_xy
        self.res = res
        self.nx, self.ny = heights.shape

    @classmethod
    def from_trimesh(cls, mesh, device: str, res: float = 0.04, z_top: float = 10.0) -> "HeightMap":
        """Raycast a terrain triangle mesh (Isaac Lab terrain generator output) straight down."""
        lo, hi = mesh.bounds[0, :2], mesh.bounds[1, :2]
        nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
        ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
        wm_mesh = wp.Mesh(points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device=device),
                          indices=wp.array(np.asarray(mesh.faces, np.int32).flatten(), dtype=wp.int32, device=device))
        out = wp.zeros(nx * ny, dtype=float, device=device)
        wp.launch(_ray_down, dim=nx * ny, inputs=[wm_mesh.id, float(lo[0]), float(lo[1]), res, ny, z_top, out],
                  device=device)
        h = wp.to_torch(out).view(nx, ny).clone()
        return cls(_fill_thin_misses(h), (float(lo[0]), float(lo[1])), res)

    def query(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear terrain height at world xy (..., 2); differentiable w.r.t. xy."""
        u = (xy[..., 0] - self.ox) / self.res
        v = (xy[..., 1] - self.oy) / self.res
        u = u.clamp(0, self.nx - 1.001)
        v = v.clamp(0, self.ny - 1.001)
        i0, j0 = u.floor().long(), v.floor().long()
        fu, fv = u - i0, v - j0
        h00, h10 = self.h[i0, j0], self.h[i0 + 1, j0]
        h01, h11 = self.h[i0, j0 + 1], self.h[i0 + 1, j0 + 1]
        return (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv

    def build_support_field(self, window: float = 0.5, hole_depth: float = 0.25, sigma: float = 0.05):
        """Smooth map in [0, 1]: 1 on supportable surface (within ``hole_depth`` of the local top surface in a
        ``window`` neighbourhood, so stair treads count but gaps between stepping stones do not), blurred by a
        Gaussian of std ``sigma`` so its gradient points toward the middle of supports over a useful band."""
        import torch.nn.functional as F

        k = max(3, int(round(window / self.res)) | 1)
        h = self.h[None, None]
        local_top = F.max_pool2d(h, k, stride=1, padding=k // 2)
        solid = (h > local_top - hole_depth).float()
        r = max(1, int(round(3 * sigma / self.res)))
        x = torch.arange(-r, r + 1, device=h.device, dtype=torch.float32) * self.res
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        s = F.conv2d(F.pad(solid, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
        s = F.conv2d(F.pad(s, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
        self.support = s[0, 0]

    def query_support(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear support-field value at world xy (differentiable)."""
        u = ((xy[..., 0] - self.ox) / self.res).clamp(0, self.nx - 1.001)
        v = ((xy[..., 1] - self.oy) / self.res).clamp(0, self.ny - 1.001)
        i0, j0 = u.floor().long(), v.floor().long()
        fu, fv = u - i0, v - j0
        s = self.support
        return (s[i0, j0] * (1 - fu) + s[i0 + 1, j0] * fu) * (1 - fv) + (s[i0, j0 + 1] * (1 - fu) + s[i0 + 1, j0 + 1] * fu) * fv

    def query_nearest(self, xy: torch.Tensor) -> torch.Tensor:
        i = ((xy[..., 0] - self.ox) / self.res).round().long().clamp(0, self.nx - 1)
        j = ((xy[..., 1] - self.oy) / self.res).round().long().clamp(0, self.ny - 1)
        return self.h[i, j]


@wp.kernel
def _ray_down(mesh: wp.uint64, x0: float, y0: float, res: float, ny: int, z_top: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    p = wp.vec3(x0 + float(tid // ny) * res, y0 + float(tid % ny) * res, z_top)
    q = wp.mesh_query_ray(mesh, p, wp.vec3(0.0, 0.0, -1.0), 2.0 * z_top)
    if q.result:
        out[tid] = z_top - q.t
    else:  # no surface below (a floorless gap, e.g. Isaac's gap terrain): deep, not flat ground (real floors reach -3 m)
        out[tid] = -5.0


def _fill_thin_misses(h: torch.Tensor, miss_value: float = -5.0) -> torch.Tensor:
    """Ray-cast misses (``miss_value``) wider than 3 cells are real floorless holes (gaps) and stay deep; thinner ones
    are slits between mesh faces (stair risers, stone floors): they take the lowest valid neighbour height."""
    import torch.nn.functional as F

    miss = (h <= miss_value + 1e-3).float()[None, None]
    big = -F.max_pool2d(-miss, 3, 1, 1)  # erode: misses with a full 3 x 3 miss neighbourhood
    big = F.max_pool2d(big, 3, 1, 1)  # dilate back: the opening of the miss mask
    thin = (miss > 0) & (big == 0)
    if not thin.any():
        return h
    hv = torch.where(miss[0, 0] > 0, torch.full_like(h, 1e3), h)[None, None]
    for _ in range(3):  # lowest valid neighbour, a few cells out
        lo = -F.max_pool2d(-hv, 3, 1, 1)
        hv = torch.where(thin & (hv >= 1e3), lo, hv)
    out = torch.where(thin[0, 0] & (hv[0, 0] < 1e3), hv[0, 0], h)
    return out
