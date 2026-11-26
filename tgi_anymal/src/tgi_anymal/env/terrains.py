"""AME2's stepping-stone terrain (Stones-Everywhere) with physical or virtual floors, as an Isaac Lab sub-terrain."""

from __future__ import annotations

import numpy as np


# AME2's training stones (Zhang et al., arXiv 2601.08485, App. A: "Stones", sparse, 30 %) follow "Stones-Everywhere" of
# Zhang et al., "Learning agile locomotion on risky terrains" (arXiv 2311.10484, Sec. III-A1 and Table I): one square
# stone per grid square, each with a uniform random x / y shift and top height; every parameter linear in the level:
#   level 0 / 5 / 9: stone size 92 / 52 / 20 cm, gap 8 / 18 / 26 cm, max shift 3.6 / 8.2 / 11.8 cm, max height 1 / 6 / 10 cm.
# AME2 floors (App. A): half the sparse tiles get a physical floor, half a virtual one (visible in the map, no collision);
# floor height uniform in [-1.5, -0.35] m. A virtual floor is marked by a 2 cm box VIRTUAL_MARK_Z below the tile corner
# (unreachable, below --term_floor), whose depth below the mark encodes the floor; ``virtual_floors`` reads them back.
STONES_EVERYWHERE = {"size": (0.92, 0.20), "gap": (0.08, 0.26), "shift": (0.036, 0.118), "height": (0.01, 0.10)}
VIRTUAL_MARK_Z = -20.0
VIRTUAL_MARK_AT = 0.05  # mark centre: tile corner + (0.05, 0.05)


def stones_everywhere_rects(level: int, size: tuple[float, float], rng: np.random.Generator, num_levels: int = 10,
                            platform_width: float = 2.0):
    """Stones-Everywhere tile at ``level`` (0 .. num_levels - 1): (stone rectangles (x0, x1, y0, y1, top_z), platform half
    width). Every stone is complete: stone size, gap and shift are scaled together (by at most pitch / 2 / tile, <= 6 %) so a
    whole number of grid cells fills the tile, the tile border runs through the middle of a gap, and since the max shift is
    below half the gap no stone crosses it; neighbouring tiles join with a normal gap. The start platform takes the place of
    the central n x n block of cells (n with the parity of the cell count, closest to ``platform_width``), so it is
    surrounded by the normal stone gap."""
    u = min(max(level, 0), num_levels - 1) / (num_levels - 1)
    P = {k: v[0] + u * (v[1] - v[0]) for k, v in STONES_EVERYWHERE.items()}
    assert size[0] == size[1], "square tiles"
    m = max(1, int(round(size[0] / (P["size"] + P["gap"]))))
    sc = size[0] / m / (P["size"] + P["gap"])  # stone size, gap and shift scaled together (|sc - 1| <= pitch / 2 / tile)
    w, g, s, h = P["size"] * sc, P["gap"] * sc, P["shift"] * sc, P["height"]
    m, pitch = [m, m], [size[0] / m] * 2
    assert s < g / 2, "shift must stay below half the gap"
    nb = []
    for k, pt in zip(m, pitch):
        cands = [n for n in range(1, k + 1) if n % 2 == k % 2]
        nb.append(min(cands, key=lambda n: abs(n * pt - g - platform_width)))
    rects = []
    for i in range(m[0]):
        for j in range(m[1]):
            if abs(i + 0.5 - m[0] / 2) <= (nb[0] - 1) / 2 + 1e-6 and abs(j + 0.5 - m[1] / 2) <= (nb[1] - 1) / 2 + 1e-6:
                continue  # inside the platform block
            x = (i + 0.5) * pitch[0] + rng.uniform(-s, s)
            y = (j + 0.5) * pitch[1] + rng.uniform(-s, s)
            rects.append((x - w / 2, x + w / 2, y - w / 2, y + w / 2, rng.uniform(-h, h)))
    return rects, min(nb[0] * pitch[0], nb[1] * pitch[1]) / 2 - (pitch[0] - w) / 2


def isaac_stones_everywhere(**kw):
    """Isaac Lab (trimesh) sub-terrain: AME2 / Stones-Everywhere stones over a physical or virtual floor, with a start
    platform at the tile centre that takes the place of a block of stones (about ``platform_width`` wide)."""
    import trimesh
    from isaaclab.terrains import SubTerrainBaseCfg
    from isaaclab.utils import configclass

    def fn(difficulty: float, cfg) -> tuple[list, np.ndarray]:
        rng = np.random.default_rng(np.random.randint(2**31))  # the generator seeds numpy per sub-terrain
        level = int(np.floor(difficulty * cfg.num_levels + 1e-6))  # difficulty = (row + U(0, 1)) / num_rows
        floor_z = -rng.uniform(*cfg.floor_depth_range)
        virtual = rng.uniform() < cfg.virtual_share
        cx, cy = cfg.size[0] / 2, cfg.size[1] / 2

        def box(x0, x1, y0, y1, z0, z1):
            return trimesh.creation.box((x1 - x0, y1 - y0, z1 - z0),
                                        trimesh.transformations.translation_matrix(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)))

        rects, a = stones_everywhere_rects(level, cfg.size, rng, cfg.num_levels, cfg.platform_width)
        meshes = [box(cx - a, cx + a, cy - a, cy + a, floor_z, 0.0)]
        if virtual:
            m = VIRTUAL_MARK_AT
            meshes.append(box(m - 0.01, m + 0.01, m - 0.01, m + 0.01, VIRTUAL_MARK_Z + floor_z - 0.01, VIRTUAL_MARK_Z + floor_z + 0.01))
        else:
            meshes.append(box(0.0, cfg.size[0], 0.0, cfg.size[1], floor_z - 0.1, floor_z))
        for x0, x1, y0, y1, top in rects:
            meshes.append(box(x0, x1, y0, y1, floor_z, top))
        return meshes, np.array([cx, cy, 0.0])

    @configclass
    class IsaacStonesEverywhereCfg(SubTerrainBaseCfg):
        function = fn
        num_levels: int = 10
        floor_depth_range: tuple[float, float] = (0.35, 1.5)
        virtual_share: float = 0.5
        platform_width: float = 2.0

    return IsaacStonesEverywhereCfg(**kw)


def virtual_floors(vertices: np.ndarray, tile: tuple[float, float]) -> list[tuple[float, float, float, float, float]]:
    """(x0, x1, y0, y1, floor_z) of every virtual-floor tile, decoded from the marks in the generated terrain mesh."""
    v = vertices[vertices[:, 2] < VIRTUAL_MARK_Z / 2]
    if len(v) == 0:
        return []
    key = np.round(v[:, :2] / 0.5).astype(np.int64)  # one mark (8 vertices) per tile, tiles >= 2 m apart
    out = []
    for k in np.unique(key, axis=0):
        c = v[(key == k).all(1)].mean(0)
        x0, y0 = c[0] - VIRTUAL_MARK_AT, c[1] - VIRTUAL_MARK_AT
        out.append((x0, x0 + tile[0], y0, y0 + tile[1], float(c[2] - VIRTUAL_MARK_Z)))
    return out
