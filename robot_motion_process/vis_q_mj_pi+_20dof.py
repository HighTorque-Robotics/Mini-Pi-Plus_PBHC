# robot_motion_process/vis_q_mj_pi+_20dof.py
import os
import sys
import time
import os.path as osp
from copy import deepcopy
from collections import defaultdict

sys.path.append(os.getcwd())

import numpy as np
import torch
import math
import joblib
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot

import hydra
from omegaconf import DictConfig, OmegaConf

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0], point1[1], point1[2],
        point2[0], point2[1], point2[2],
    )


# globals manipulated by the key callback
curr_start = num_motions = motion_id = 0
motion_acc = set()
time_step = 0.0
dt = 1 / 30
speed = 1.0
paused = False
rewind = False
motion_data_keys = []
contact_mask = None
curr_time = 0
resave = False


def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    try:
        ch = chr(keycode)
    except Exception:
        ch = ""

    if ch == "R":
        print("Reset")
        time_step = 0
    elif ch == " ":
        print("Paused")
        paused = not paused
    elif keycode == 256 or ch == "Q":
        print("Esc")
        os._exit(0)
    elif ch == "L":
        speed = speed * 1.5
        print("Speed: ", speed)
    elif ch == "K":
        speed = speed / 1.5
        print("Speed: ", speed)
    elif ch == "J":
        print("Toggle Rewind: ", not rewind)
        rewind = not rewind
    elif keycode == 262:  # Right
        time_step += dt
    elif keycode == 263:  # Left
        time_step -= dt
    elif ch == "Q":
        if contact_mask is not None:
            print("Modify left foot contact!!!")
            contact_mask[curr_time][0] = 1.0 - contact_mask[curr_time][0]
            resave = True
    elif ch == "E":
        if contact_mask is not None:
            print("Modify right foot contact!!!")
            contact_mask[curr_time][1] = 1.0 - contact_mask[curr_time][1]
            resave = True
    else:
        print("not mapped", ch, keycode)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave

    # init globals
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind = \
        0, 1, 0, set(), 0.0, 1 / 30, 1.0, False, False
    curr_time = 0
    resave = False

    # ---- load motion file
    motion_file = cfg.motion_file
    motions_loaded = joblib.load(motion_file)

    # ---- normalize into dict-of-motions
    if isinstance(motions_loaded, dict) and {"dof", "root_rot"}.issubset(motions_loaded.keys()):
        motions = {"motion": motions_loaded}  # single motion dict wrapped
    elif isinstance(motions_loaded, dict):
        motions = motions_loaded  # already dict-of-motions
    else:
        raise TypeError(f"Unsupported motion file format: {type(motions_loaded)}")

    motion_keys = list(motions.keys())
    motion_id = 0
    curr_motion_key = motion_keys[motion_id]
    curr_motion = motions[curr_motion_key]

    print(motion_file)

    # ---- timing params
    speed = 1.0 if "speed" not in cfg else cfg.speed
    hang = False if "hang" not in cfg else cfg.hang
    dt = 1.0 / float(curr_motion["fps"]) if isinstance(curr_motion, dict) and "fps" in curr_motion \
        else (cfg.dt if "dt" in cfg else 1 / 30)

    # ---- ensure root translation key
    if "root_trans_offset" not in curr_motion:
        if "root_pos" in curr_motion:
            curr_motion["root_trans_offset"] = curr_motion["root_pos"]
        else:
            raise KeyError("Neither 'root_trans_offset' nor 'root_pos' found in motion data.")

    print("Motion file: ", motion_file)
    print("Motion length: ", curr_motion["dof"].shape[0], "frames")
    print("Speed: ", speed)
    print()

    # ---- optional visual overlays
    if "contact_mask" in curr_motion.keys():
        contact_mask = curr_motion["contact_mask"]
    else:
        contact_mask = None

    # ---- model path
    humanoid_xml = "./description/robots/pi+_all/pi_plus_20dof_250828/xml/pi_20dof_0828.xml"
    print(humanoid_xml)

    vis_smpl = False if "vis_smpl" not in cfg else cfg.vis_smpl
    vis_tau_key = "tau" if "vis_tau_key" not in cfg else cfg.vis_tau_key
    vis_tau = (vis_tau_key in curr_motion) if "vis_tau" not in cfg else cfg.vis_tau
    vis_contact = ("contact_mask" in curr_motion) if "vis_contact" not in cfg else cfg.vis_contact

    if vis_smpl:
        assert "smpl_joints" in curr_motion, "vis_smpl=True requires 'smpl_joints' in motion."
    if vis_tau:
        assert vis_tau_key in curr_motion and not vis_contact, "vis_tau requires tau present and vis_contact=False."
    if vis_contact:
        assert "contact_mask" in curr_motion and not vis_tau, "vis_contact requires contact_mask and vis_tau=False."

    # ---- FK for joints if not visualizing SMPL joints
    if not vis_smpl:
        cfg_robot = OmegaConf.load("./description/robots/cfg/robot/pi+_20dof.yaml")
        humanoid_fk = Humanoid_Batch(cfg_robot)
        pose_aa = torch.from_numpy(curr_motion["pose_aa"]).unsqueeze(0)
        root_trans = torch.from_numpy(curr_motion["root_trans_offset"]).unsqueeze(0)
        fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)
        joint_gt = fk_return.global_translation_extend[0]

    # ---- MuJoCo init
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    # debug: list joints
    print("\n=== Joint Names ===")
    for i in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"Joint {i}: {joint_name}")
    print("==================\n")

    # debug: list bodies
    print("=== Link/Body Names ===")
    for i in range(mj_model.nbody):
        link_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Link {i}: {link_name}")
    print("==================\n")

    print(
        "Init Pose: ",
        (
            np.array(
                np.concatenate(
                    [
                        curr_motion["root_trans_offset"][0],
                        curr_motion["root_rot"][0][[3, 0, 1, 2]],  # xyzw -> wxyz
                        curr_motion["dof"][0],
                    ]
                ),
                dtype=np.float32,
            ).__repr__()
        ),
    )

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        viewer.cam.lookat[:] = np.array([0, 0, 0.7])
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30

        # draw some markers
        for _ in range(50):
            add_visual_capsule(
                viewer.user_scn,
                np.zeros(3),
                np.array([0.001, 0, 0]),
                0.03,
                np.array([1, 0, 0, 1]),
            )

        while viewer.is_running():
            step_start = time.time()

            total_T = curr_motion["dof"].shape[0] * dt
            if time_step >= total_T:
                time_step -= total_T
            if time_step < 0:
                time_step += total_T

            curr_time = int(round(time_step / dt)) % curr_motion["dof"].shape[0]

            if hang:
                mj_data.qpos[:3] = np.array([0, 0, 0.8])
            else:
                mj_data.qpos[:3] = curr_motion["root_trans_offset"][curr_time]
            mj_data.qpos[3:7] = curr_motion["root_rot"][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion["dof"][curr_time]

            mujoco.mj_forward(mj_model, mj_data)

            if not paused:
                time_step += dt * (1 if not rewind else -1) * speed

            if vis_smpl:
                joint_gt = curr_motion["smpl_joints"]
                if not np.all(joint_gt[curr_time] == 0):
                    for i in range(joint_gt.shape[1]):
                        viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
            else:
                # skip geom[0] which we used as an origin marker; place joints starting at 1
                for i in range(min(21, joint_gt.shape[1] - 1)):
                    viewer.user_scn.geoms[i + 1].pos = joint_gt[curr_time, i + 1]

            if vis_contact and contact_mask is not None:
                # indices 5 and 15 are arbitrary here; adjust if your markers differ
                viewer.user_scn.geoms[5].rgba = np.array([0, 1 - contact_mask[curr_time, 0], 0, 1])
                viewer.user_scn.geoms[15].rgba = np.array([0, 1 - contact_mask[curr_time, 1], 0, 1])

            print(
                "Init Pose: ",
                (
                    np.array(
                        np.concatenate(
                            [
                                curr_motion["root_trans_offset"][0],
                                curr_motion["root_rot"][0][[3, 0, 1, 2]],
                                curr_motion["dof"][0],
                            ]
                        ),
                        dtype=np.float32,
                    ).__repr__()
                ),
            )
            print("curr_time: ", curr_time)

            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            print("Frame ID: ", curr_time, "\t | Time ", f"{time_step:4f}", end="\r\b")

    # optional re-save when edited in viewer
    if resave:
        motions[curr_motion_key]["contact_mask"] = contact_mask
        out_file = motion_file.split(".")[0] + "_edit_cont.pkl"
        print("Saving edited contacts to:", out_file)
        joblib.dump(motions, out_file)


if __name__ == "__main__":
    main()