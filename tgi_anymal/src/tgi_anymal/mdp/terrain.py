"""Terrain features for the world model and the critic: base height, base and foot scans from the heightmap, and the
clearance, surface normal and contact geometry of the collision spheres and feet from the terrain distance field.

Each foot (a sphere) gets a small footprint grid around its contact point, aligned with the base yaw.
"""

from __future__ import annotations

import torch

from .heightmap import HeightMap
from .rotations import yaw_mat

from ..constants import FOOT_RADIUS, NUM_CB, NUM_FEET

NOMINAL_HEIGHT = 0.6


class TerrainFeatures:
    FOOT_X = (-0.03, 0.03, 3)  # footprint of the 3.1 cm foot sphere
    FOOT_Y = (-0.03, 0.03, 3)
    BASE_X = (-0.7, 0.9, 9)
    BASE_Y = (-0.45, 0.45, 5)
    CRITIC_X = (-0.8, 1.0, 10)
    CRITIC_Y = (-0.5, 0.5, 5)

    def __init__(self, hm: HeightMap, device: str):
        self.hm = hm
        self.foot_grid = self._rect(self.FOOT_X, self.FOOT_Y, device)
        self.base_grid = self._rect(self.BASE_X, self.BASE_Y, device)
        self.foot_center = (self.FOOT_X[2] // 2) * self.FOOT_Y[2] + self.FOOT_Y[2] // 2
        self.critic_grid = self._rect(self.CRITIC_X, self.CRITIC_Y, device)
        nf, nb = self.foot_grid.shape[0], self.base_grid.shape[0]
        self.wm_dim = 1 + nb + NUM_FEET * (1 + nf)
        self.critic_dim = self.critic_grid.shape[0]
        self.field = None  # set by add_contacts: the body's collision spheres vs. the terrain
        self.col_radius = self.col_margin = self.col_weight = self.term_offset = None
        self.col_roles = []

    # what touching the terrain means for a body point (collision sphere): margin of the proximity penalty (m), its
    # weight, and whether a touch ends the episode
    ROLES = {"allowed": (0.03, 0.0, False), "tolerated": (0.03, 1.0, False), "forbidden": (0.05, 1.0, True)}

    def add_contacts(self, field, col_radius: torch.Tensor, col_names: list[str], roles: str):
        """The world model also gets, for every collision sphere of the body (thighs, knees, shanks, belly), its
        clearance to the terrain and the nearest surface's normal (heading frame): the contacts of the legs and base
        with box edges and walls are in its input, not only the feet. roles: "part:role,..." over the sphere names
        (later entries win; default forbidden)."""
        self.field = field
        dev = col_radius.device
        role = ["forbidden"] * len(col_names)
        for part, r in (x.split(":") for x in roles.split(",") if x):
            assert r in self.ROLES, r
            role = [r if part in n else old for n, old in zip(col_names, role)]
        self.col_roles = role
        self.col_radius = col_radius
        self.col_margin = torch.tensor([self.ROLES[r][0] for r in role], dtype=torch.float32, device=dev)
        self.col_weight = torch.tensor([self.ROLES[r][1] for r in role], dtype=torch.float32, device=dev)
        self.term_offset = torch.tensor([0.0 if self.ROLES[r][2] else 1.0 for r in role], device=dev)
        self.wm_dim += 4 * len(col_names)
        # contact bodies of the world model (constants.CONTACT_BODIES): feet, shanks, thighs (thigh + knee capsules),
        # base; mask (bodies after the feet, spheres)
        legs = ("LF", "RF", "LH", "RH")
        grp = []
        for n in col_names:
            if n[:2] in legs and "shank" in n:
                grp.append(NUM_FEET + legs.index(n[:2]))
            elif n[:2] in legs and ("thigh" in n or "knee" in n):
                grp.append(2 * NUM_FEET + legs.index(n[:2]))
            else:
                assert n.startswith("base"), n
                grp.append(NUM_CB - 1)
        g = torch.tensor(grp, device=dev)
        self.sphere_group = torch.stack([(g == k).float() for k in range(NUM_FEET, NUM_CB)])  # (NUM_CB - 4, n_col)
        assert bool((self.sphere_group.sum(-1) > 0).all()), grp

    def clearance(self, cp) -> torch.Tensor:
        return self.field.query(cp) - self.col_radius

    @staticmethod
    def _rect(xr, yr, device) -> torch.Tensor:
        xs = torch.linspace(*xr, device=device)
        ys = torch.linspace(*yr, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((gx.flatten(), gy.flatten()), -1)

    @staticmethod
    def _yaw2(R: torch.Tensor) -> torch.Tensor:
        return yaw_mat(R)[..., :2, :2]

    def _points(self, center_xy, R2, grid):
        # center (..., 2), R2 (..., 2, 2), grid (G, 2) -> (..., G, 2) world xy
        return center_xy.unsqueeze(-2) + (R2.unsqueeze(-3) @ grid.unsqueeze(-1)).squeeze(-1)

    def wm_features(self, p, R, fp, cp):
        """fp: foot contact points (..., n_feet, 3) (sphere centre minus radius); cp: collision sphere centres
        (..., n_col, 3).

        Returns (features, foot clearance (..., n_feet), terrain roughness under each foot (..., n_feet), unsupported
        footprint share and 1 - support field under each foot (..., n_feet, 2), collision sphere clearances
        (..., n_col), contact geometry (``contact_geometry``)).
        """
        R2 = self._yaw2(R)
        h0 = self.hm.query(p[..., :2])
        hb = (p[..., 2] - h0).unsqueeze(-1).clamp(-1.0, 2.0)  # over a floorless gap the column top is -5 m
        # relative heights clamped to +-1 m
        base = (self.hm.query(self._points(p[..., :2], R2, self.base_grid)) - h0.unsqueeze(-1)).clamp(-1.0, 1.0)
        pts = self._points(fp[..., :2], R2.unsqueeze(-3).expand(*fp.shape[:-1], 2, 2), self.foot_grid)  # (..., n_feet, 9, 2)
        h = self.hm.query(pts)
        h_c = h[..., self.foot_center]
        clear = fp[..., 2] - h_c
        rel = (h - h_c.unsqueeze(-1)).clamp(-1.0, 1.0)
        rough = rel.std(-1)
        # unsupported fraction of the footprint: grid points more than ~3 cm off the foot's support height; exact
        # (nearest-node) heights forward, bilinear gradient (bilinear values smear stone edges over a cell)
        h_n = self.hm.query_nearest(pts)
        rel_s = rel + ((h_n - h_n[..., self.foot_center].unsqueeze(-1)).clamp(-1.0, 1.0) - rel).detach()
        unsupported = torch.sigmoid((rel_s.abs() - 0.03) / 0.01).mean(-1)
        # smooth support field: its gradient pulls the feet toward stone centres
        unsupported_field = 1.0 - self.hm.query_support(pts).mean(-1)
        unsupported = torch.stack((unsupported, unsupported_field), -1)
        feats = torch.cat((hb, base, clear.clamp(-0.5, 1.0), rel.flatten(-2)), -1)  # (gaps: column tops at -5 m)
        # collision spheres: clearance to the terrain field and the nearest surface's normal
        dc, nc = self.field.query(cp, normal=True)
        clr = dc - self.col_radius
        dist = clr.clamp(-0.1, 0.3)
        Ht = yaw_mat(R).transpose(-1, -2)  # normals in the heading frame (the model has no heading); 0 when far
        nrm = (Ht.unsqueeze(-3) @ nc.unsqueeze(-1)).squeeze(-1) * (dist.detach() < 0.3).unsqueeze(-1)
        feats = torch.cat((feats, dist, nrm.flatten(-2)), -1)
        geo = self.contact_geometry(fp, cp, clr, nc)
        return feats, clear, rough, unsupported, clr, geo

    def contact_geometry(self, fp, cp, clr, nc) -> dict:
        """Exact contact geometry of each contact body (constants.CONTACT_BODIES) on the 3D field: its gap to the
        terrain phi (..., NUM_CB) (nearest sphere of the body; the foot sphere for the feet), the surface normal
        n (..., NUM_CB, 3) (world, no gradient) and the contact point x (..., NUM_CB, 3) (where a force would act)."""
        up = torch.tensor((0.0, 0.0, FOOT_RADIUS), device=fp.device)
        fc = fp + up  # foot sphere centres
        df, nf = self.field.query(fc, normal=True)
        phi_f, x_f = df - FOOT_RADIUS, fc - FOOT_RADIUS * nf
        big = 10.0 * (1.0 - self.sphere_group)  # (G, n_col): other bodies' spheres out of the running
        cl = clr.unsqueeze(-2) + big
        idx = cl.detach().argmin(-1)  # (..., G) nearest sphere per body
        phi_b = cl.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        gi = idx.unsqueeze(-1).expand(*idx.shape, 3)
        n_b = nc.gather(-2, gi)
        r_b = self.col_radius[idx]
        x_b = cp.gather(-2, gi) - r_b.unsqueeze(-1) * n_b
        return {"phi": torch.cat((phi_f, phi_b), -1), "n": torch.cat((nf, n_b), -2), "x": torch.cat((x_f, x_b), -2)}

    def critic_scan(self, p, R) -> torch.Tensor:
        R2 = self._yaw2(R)
        pts = self._points(p[..., :2], R2, self.critic_grid)
        # bilinear, so the value gradient reaches the base position
        return (p[..., 2, None] - self.hm.query(pts) - NOMINAL_HEIGHT).clamp(-1.0, 1.0)
