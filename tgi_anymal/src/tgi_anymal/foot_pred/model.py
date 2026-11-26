"""TGI's continually trained FPM: the FPM network (``fpm.foot_pred``) with a running input normalizer.

Input: the ``system_state``; output: the next touchdown (x, y) of each foot in the base frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fpm.foot_pred import foot_predictor_mlp

from ..constants import NUM_FEET
from ..mdp.interface import SYSTEM_STATE_DIM
from ..world_model.networks import RunningNorm


class FootholdPredictor(nn.Module):
    def __init__(self, in_dim: int = SYSTEM_STATE_DIM, hidden=(128, 128)):
        super().__init__()
        self.in_dim, self.hidden = in_dim, tuple(hidden)
        self.net = foot_predictor_mlp(in_dim, 2 * NUM_FEET, self.hidden)
        self.norm = RunningNorm(in_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., n_feet, 2) footholds in the root frame."""
        return self.net(self.norm(state)).view(*state.shape[:-1], NUM_FEET, 2)
