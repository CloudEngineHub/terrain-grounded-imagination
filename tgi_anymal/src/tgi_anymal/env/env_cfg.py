"""ANYmal-D env for world-model training on Newton: Isaac Lab 3's ANYmal-D rough velocity task with the ANYdrive LSTM
actuator model, the FPM training terrains plus stepping stones, a pit and parkour obstacles, and lagged actions (domain
randomization is applied by ``sim.py``).
"""

from __future__ import annotations

import numpy as np
import torch



def isaac_terrains_cfg(stones_share: float = 0.3, num_cols: int = 30, pit_step: float = 0.0,
                       parkour: dict[str, int] | None = None):
    """The FPM training terrains (``fpm.terrains``), with stepping stones on ``stones_share`` of the columns.

    Columns are assigned explicitly so that every terrain type gets at least one; ``pit_step`` > 0 adds a
    step-out-of-a-pit terrain (5 to 40 cm deep) on one extra column; ``parkour`` adds columns of climb-up, climb-down
    and gap obstacles.
    """
    from fpm.terrains import TRAINING_TERRAINS_CFG

    from .terrains import isaac_stones_everywhere

    cfg = TRAINING_TERRAINS_CFG.copy()
    subs = dict(cfg.sub_terrains)
    # AME2's training stones (Stones-Everywhere, physical / virtual floors) replace both stone types
    subs.pop("stepping_stones_flat", None)
    subs["stepping_stones"] = isaac_stones_everywhere()
    if pit_step > 0:
        from isaaclab.terrains import MeshPitTerrainCfg

        subs["pit_step"] = MeshPitTerrainCfg(proportion=pit_step, pit_depth_range=(0.05, 0.40), platform_width=2.0)
    n_stones = round(stones_share * num_cols)
    num_cols += pit_step > 0  # the pit gets its own extra column, the other types keep theirs
    others = [k for k in subs if k != "stepping_stones"]
    base = TRAINING_TERRAINS_CFG.sub_terrains
    w = np.array([base[k].proportion if k in base else subs[k].proportion for k in others])
    free = num_cols - n_stones - len(others)  # one column each first
    share = w / w.sum() * free
    cols = 1 + np.floor(share).astype(int)
    for i in np.argsort(-(share - np.floor(share)))[: free - int(np.floor(share).sum())]:
        cols[i] += 1
    for k, c in zip(others, cols):
        subs[k].proportion = float(c) / num_cols
    subs["stepping_stones"].proportion = n_stones / num_cols
    if parkour:  # extra columns; the other types keep theirs
        from isaaclab.terrains import MeshBoxTerrainCfg, MeshGapTerrainCfg, MeshPitTerrainCfg

        fams = {  # ANYmal Parkour's climb-up / climb-down / jump obstacles, easiest to hardest over the rows
            "climb_up": lambda: MeshPitTerrainCfg(pit_depth_range=(0.1, 1.0), platform_width=3.0),
            "climb_down": lambda: MeshBoxTerrainCfg(box_height_range=(0.1, 1.0), platform_width=2.0),
            "gap": lambda: MeshGapTerrainCfg(gap_width_range=(0.1, 1.0), platform_width=2.0),  # ANYmal Parkour: up to 1 m
        }
        total = num_cols + sum(parkour.values())
        for v in subs.values():
            v.proportion *= num_cols / total
        for fam, n in parkour.items():
            subs[fam] = fams[fam]()
            subs[fam].proportion = n / total
        num_cols = total
    cfg.sub_terrains = subs
    cfg.num_cols = num_cols
    return cfg


def make_isaac_env_cfg(num_envs: int, seed: int = 0, stones_share: float = 0.3, episode_s: float = 8.0,
                       pit_step: float = 0.0, parkour: dict[str, int] | None = None):
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
    import isaaclab.envs.mdp as imdp

    cfg = AnymalDRoughEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = episode_s
    cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp
    cfg.scene.height_scanner.pattern_cfg.resolution = 0.05  # hires height map
    # contact force history over the whole control step: the world model's labels are the step-mean force
    cfg.scene.contact_forces.history_length = max(int(cfg.decimation), cfg.scene.contact_forces.history_length)
    gen = isaac_terrains_cfg(stones_share, pit_step=pit_step, parkour=parkour)
    gen.curriculum = True  # rows = difficulty levels (the success-guided sampler picks the cells)
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    for k in ("physics_material", "add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
        setattr(cfg.events, k, None)

    @configclass
    class _Obs:
        @configclass
        class _Policy(ObservationGroupCfg):  # the pipeline builds its own observations from the state
            base_lin_vel = ObservationTermCfg(func=imdp.base_lin_vel)

        policy: _Policy = _Policy()

    cfg.observations = _Obs()
    cfg.actions.joint_pos = JointPositionActionCfg(class_type=_lagged_action_cls(), asset_name="robot",
                                                   joint_names=[".*"], scale=0.5, use_default_offset=True)
    cfg.terminations.wm_extra = TerminationTermCfg(func=_extra_termination)
    return cfg


def _extra_termination(env) -> torch.Tensor:
    sim = getattr(env, "_loco_sim", None)
    if sim is None or sim.extra_termination is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    from ..mdp.mdp import unflatten_state

    return sim.extra_termination(unflatten_state(sim.flat_state()))


def _lagged_action_cls():
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    class _Lagged(JointPositionAction):
        """Joint targets reach the actuators ``lag`` physics steps late (per env), as mjlab's actuator delay."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.lag = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._prev = self._offset.expand(env.num_envs, -1).clone()
            self._k = 0

        def process_actions(self, actions):
            self._prev = self.processed_actions.clone()
            super().process_actions(actions)
            self._k = 0

        def apply_actions(self):
            tgt = torch.where((self._k < self.lag).unsqueeze(-1), self._prev, self.processed_actions)
            self._asset.set_joint_position_target_index(target=tgt, joint_ids=self._joint_ids)
            self._k += 1

        def reset(self, env_ids=None):
            super().reset(env_ids)
            ids = slice(None) if env_ids is None else env_ids
            self._prev[ids] = self._offset.expand(self.num_envs, -1)[ids]
            self._processed_actions[ids] = self._offset.expand(self.num_envs, -1)[ids]

    return _Lagged
