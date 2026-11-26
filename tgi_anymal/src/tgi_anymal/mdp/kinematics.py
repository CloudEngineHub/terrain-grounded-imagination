"""Differentiable forward kinematics from the MuJoCo model: foot sites, selected bodies, collision spheres and hinge
frames."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import quat_to_mat, skew


class Kinematics:
    def __init__(self, model: mujoco.MjModel, joint_names: list[str], device: str, site_names, body_names):
        self.device = device
        m = model
        jid = {}
        for i in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            jid[nm] = jid[nm.split("/")[-1]] = i
        # joint index (in our action/state ordering) driving each body
        self.body_joint = {}
        for k, name in enumerate(joint_names):
            j = jid[name]
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            assert np.allclose(m.jnt_pos[j], 0.0), name
            self.body_joint[int(m.jnt_bodyid[j])] = (k, torch.tensor(m.jnt_axis[j], dtype=torch.float32, device=device))
        root = int(m.jnt_bodyid[[i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]])
        self.root = root
        self.model_parent = np.array(m.body_parentid)

        def bid(n):
            for i in range(m.nbody):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                if nm == n or (nm and nm.endswith("/" + n)):
                    return i
            raise KeyError(n)

        def sid(n):
            for i in range(m.nsite):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
                if nm == n or (nm and nm.endswith("/" + n)):
                    return i
            raise KeyError(n)

        targets = []  # (body id, local offset)
        for s in site_names:
            i = sid(s)
            targets.append((int(m.site_bodyid[i]), torch.tensor(m.site_pos[i], dtype=torch.float32, device=device)))
        for b in body_names:
            targets.append((bid(b), torch.zeros(3, device=device)))
        self.n_sites = len(site_names)
        self.targets = targets
        # chain of bodies root(excluded) -> target, for each target
        self.chains = []
        for b, _ in targets:
            chain = []
            while b != root:
                chain.append(b)
                b = int(m.body_parentid[b])
            self.chains.append(chain[::-1])
        self.body_pos = {b: torch.tensor(m.body_pos[b], dtype=torch.float32, device=device) for c in self.chains for b in c}
        self.body_R = {
            b: quat_to_mat(torch.tensor(m.body_quat[b], dtype=torch.float32, device=device)) for c in self.chains for b in c
        }
        # collision spheres: the model's named sphere / capsule colliders (thighs, knees, shanks, belly), each capsule as
        # spheres at its ends and (if long enough) its middle. The feet are the foot sites. For the distance to the
        # terrain field.
        self.col = []  # (body id, offsets (n, 3))
        per_body = {}
        for g in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "collision" not in gname or "foot" in gname:
                continue
            gtype, size = int(m.geom_type[g]), m.geom_size[g]
            pos = np.array(m.geom_pos[g], dtype=np.float64)
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                pts = [pos]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                ax = R.reshape(3, 3)[:, 2] * size[1]
                pts = [pos - ax, pos + ax] + ([pos] if size[1] > 0.6 * size[0] else [])
            else:
                continue
            b = int(m.geom_bodyid[g])
            for pnt in pts:
                per_body.setdefault(b, []).append(pnt)
        for b, pts in per_body.items():
            self.col.append((b, torch.tensor(np.stack(pts), dtype=torch.float32, device=device)))
        # radii / names in the per-body order of self.col
        rr, nn = [], []
        seq = {}
        for g in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "collision" not in gname or "foot" in gname:
                continue
            gtype, size = int(m.geom_type[g]), m.geom_size[g]
            n_pts = 1 if gtype == mujoco.mjtGeom.mjGEOM_SPHERE else (3 if size[1] > 0.6 * size[0] else 2)
            seq.setdefault(int(m.geom_bodyid[g]), []).extend([(float(size[0]), gname.replace("_collision", ""))] * n_pts)
        for b, _ in self.col:
            for r, n in seq[b]:
                rr.append(r), nn.append(n)
        self.col_radius = torch.tensor(rr, dtype=torch.float32, device=device)
        self.col_names = nn
        self.n_col = len(rr)
        self.col_chains = []
        for b, _ in self.col:
            chain = []
            while b != root:
                chain.append(b)
                b = int(m.body_parentid[b])
            self.col_chains.append(chain[::-1])
        for c in self.col_chains:
            for b in c:
                if b not in self.body_pos:
                    self.body_pos[b] = torch.tensor(m.body_pos[b], dtype=torch.float32, device=device)
                    self.body_R[b] = quat_to_mat(torch.tensor(m.body_quat[b], dtype=torch.float32, device=device))
        lo, hi = [], []
        for name in joint_names:
            j = jid[name]
            lo.append(m.jnt_range[j][0]), hi.append(m.jnt_range[j][1])
        self.q_lo = torch.tensor(lo, device=device)
        self.q_hi = torch.tensor(hi, device=device)
        # every hinge's body chain (for the joint frames of J^T F) and, per named body, which joints move it
        self.joint_body = {k: b for b, (k, _) in self.body_joint.items()}
        self.n_joints = len(joint_names)
        self.jchains = []
        for k in range(self.n_joints):
            b, chain = self.joint_body[k], []
            while b != root:
                chain.append(b)
                b = int(m.body_parentid[b])
            self.jchains.append(chain[::-1])
            for bb in chain:
                if bb not in self.body_pos:
                    self.body_pos[bb] = torch.tensor(m.body_pos[bb], dtype=torch.float32, device=device)
                    self.body_R[bb] = quat_to_mat(torch.tensor(m.body_quat[bb], dtype=torch.float32, device=device))
        self._bid = bid

    def joint_mask(self, body_names) -> torch.Tensor:
        """(n_bodies, n_joints): 1 where the joint moves the named body (it is on the body's chain from the base)."""
        mask = torch.zeros(len(body_names), self.n_joints, device=self.device)
        for i, n in enumerate(body_names):
            b = self._bid(n)
            while b != self.root:
                if b in self.body_joint:
                    mask[i, self.body_joint[b][0]] = 1.0
                b = int(self.model_parent[b])
        return mask

    def joint_frames(self, p, R, q):
        """World axis and origin of every hinge (..., n_joints, 3) each (the columns of the contact Jacobian)."""
        cache, axes, orgs = {}, [None] * self.n_joints, [None] * self.n_joints
        for k, chain in enumerate(self.jchains):
            Rb, pb = R, p
            for b in chain:
                if b in cache:
                    Rb, pb, a_w = cache[b]
                    continue
                pb = pb + (Rb @ self.body_pos[b].unsqueeze(-1)).squeeze(-1)
                Rb = Rb @ self.body_R[b]
                a_w = None
                if b in self.body_joint:
                    kk, ax = self.body_joint[b]
                    a_w = (Rb @ ax.unsqueeze(-1)).squeeze(-1)
                    Rb = Rb @ self._hinge(ax, q[..., kk])
                cache[b] = (Rb, pb, a_w)
            axes[k], orgs[k] = cache[chain[-1]][2], cache[chain[-1]][1]
        return torch.stack(axes, -2), torch.stack(orgs, -2)

    @staticmethod
    def _hinge(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        K = skew(axis)
        s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
        eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
        return eye + s * K + (1 - c) * (K @ K)

    def forward(self, p, R, v, w, q, qd, collision: bool = False):
        """Returns site positions/velocities (..., n_sites, 3), body rotations (..., n_bodies, 3, 3), body positions;
        with ``collision`` also the collision sphere centres (..., n_col, 3) (radii: ``col_radius``)."""
        cache = {}
        pos_out, vel_out, rot_out, bpos_out = [], [], [], []
        for t, ((b_t, off), chain) in enumerate(zip(self.targets, self.chains)):
            Rb, pb = R, p
            axes_w, origins = [], []
            joints = []
            for b in chain:
                if b in cache:
                    Rb, pb, axes_w, origins, joints = cache[b]
                    continue
                pb = pb + (Rb @ self.body_pos[b].unsqueeze(-1)).squeeze(-1)
                Rb = Rb @ self.body_R[b]
                axes_w, origins, joints = list(axes_w), list(origins), list(joints)
                if b in self.body_joint:
                    k, ax = self.body_joint[b]
                    axes_w.append((Rb @ ax.unsqueeze(-1)).squeeze(-1))
                    origins.append(pb)
                    joints.append(k)
                    Rb = Rb @ self._hinge(ax, q[..., k])
                cache[b] = (Rb, pb, axes_w, origins, joints)
            x = pb + (Rb @ off.unsqueeze(-1)).squeeze(-1)
            if t < self.n_sites:
                vel = v + torch.cross(w, x - p, dim=-1)
                for a, o, k in zip(axes_w, origins, joints):
                    vel = vel + qd[..., k : k + 1] * torch.cross(a, x - o, dim=-1)
                pos_out.append(x)
                vel_out.append(vel)
            else:
                rot_out.append(Rb)
                bpos_out.append(x)
        out = (torch.stack(pos_out, -2), torch.stack(vel_out, -2), torch.stack(rot_out, -3), torch.stack(bpos_out, -2))
        if not collision:
            return out
        cps = []
        for (b_t, offs), chain in zip(self.col, self.col_chains):
            Rb, pb = R, p
            for b in chain:
                if b in cache:
                    Rb, pb = cache[b][0], cache[b][1]
                    continue
                pb = pb + (Rb @ self.body_pos[b].unsqueeze(-1)).squeeze(-1)
                Rb = Rb @ self.body_R[b]
                if b in self.body_joint:
                    k, ax = self.body_joint[b]
                    Rb = Rb @ self._hinge(ax, q[..., k])
                cache[b] = (Rb, pb, [], [], [])
            cps.append(pb.unsqueeze(-2) + (Rb.unsqueeze(-3) @ offs.unsqueeze(-1)).squeeze(-1))
        return out + (torch.cat(cps, -2),)
