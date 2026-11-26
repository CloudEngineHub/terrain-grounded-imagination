"""Constants shared by the training simulator, imagination and evaluation."""

import os

NUM_J = 12
NUM_FEET = 4
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J  # p, quat(wxyz), v_w, w_w, q, qd
# foot friction, base com offset(3), encoder bias(12), joint friction loss, joint damping, armature scale,
# actuation lag (physics steps), base added mass
XI_DIM = 1 + 3 + NUM_J + 5
MAX_LAG = 4  # physics steps of 5 ms -> up to 20 ms actuation latency

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# ANYmal-D MJCF for the differentiable kinematics (from the official URDF, no meshes), with a site at each foot-sphere
# centre and named collision capsules / spheres (thighs, knees, shanks, belly) for the distances to the terrain
KIN_MJCF = os.path.join(ASSETS, "anymal_d_kinematics_col.xml")
FOOT_SITES = ("LF_FOOT_site", "RF_FOOT_site", "LH_FOOT_site", "RH_FOOT_site")  # foot-sphere centres
FOOT_BODIES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
FOOT_RADIUS = 0.0315
BASE_BODY = "base"
# the bodies whose net contact force the simulator reports and the world model predicts, feet first (the feet's
# loaded probabilities are the foot contacts of rewards and critic)
CONTACT_BODIES = FOOT_BODIES + ("LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK", "LF_THIGH", "RF_THIGH", "LH_THIGH",
                                "RH_THIGH", "base")
NUM_CB = len(CONTACT_BODIES)
# torque proxy for the dof_torques penalty in imagination (the world model does not predict torques): a PD law with
# typical ANYmal joint gains, clipped at the ANYdrive effort limit
KP, KD, EFFORT_LIMIT = 80.0, 2.0, 80.0

