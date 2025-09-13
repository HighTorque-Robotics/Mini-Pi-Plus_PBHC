
import os
import sys
import time
import numpy as np
import mujoco, mujoco_viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
import csv
from pathlib import Path
import onnxruntime as ort
from pynput import keyboard
import argparse
import signal
import math
from tqdm import tqdm
import joblib
import json
from typing import Optional

def load_rel_fut_ref_motion_state_flat_from_json(json_file_path):
    """
    Load rel_fut_ref_motion_state_flat data from JSON file
    
    Args:
        json_file_path (str): JSON file path
        
    Returns:
        numpy.ndarray: Data array with shape [num_steps, data_dim]
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file does not exist: {json_file_path}")
    
    print(f"Loading rel_fut_ref_motion_state_flat data from JSON file: {json_file_path}")
    
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    # Check JSON data format
    if 'raw_data' in json_data:
        # Use simplified format
        raw_data = json_data['raw_data']
        data_array = np.array(raw_data, dtype=np.float32)
        print(f"Using simplified format, loaded data with shape {data_array.shape}")
    elif 'data' in json_data:
        # Use detailed format
        data_list = []
        for step_data in json_data['data']:
            if step_data['values'] is not None:
                data_list.append(step_data['values'])
            else:
                # If data for a step is None, use zero vector
                if len(data_list) > 0:
                    data_list.append([0.0] * len(data_list[0]))
        
        data_array = np.array(data_list, dtype=np.float32)
        print(f"Using detailed format, loaded data with shape {data_array.shape}")
    else:
        raise ValueError("Incorrect JSON file format, missing 'raw_data' or 'data' field")
    
    # Validate data dimensions
    if len(data_array.shape) != 2:
        raise ValueError(f"Incorrect data dimension, expected 2D, got {len(data_array.shape)}D")
    
    if data_array.shape[1] != 406:
        print(f"Warning: Data dimension {data_array.shape[1]} does not equal expected 406 dimensions")
    
    print(f"Successfully loaded rel_fut_ref_motion_state_flat data: {data_array.shape}")
    return data_array


class ONNXPolicyWrapper:
    """ONNX model inference wrapper"""
    def __init__(self, onnx_model_path):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"ONNX Model loaded successfully")
        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
        
    def __call__(self, obs_array):
        if hasattr(obs_array, 'numpy'):
            obs_array = obs_array.numpy()
        
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        
        # Run inference
        result = self.session.run([self.output_name], {self.input_name: obs_array})
        return result[0]

def get_gravity_orientation(quat):
    r = R.from_quat(quat)
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return r.apply(gravity_vec, inverse=True)


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)

    return (qpos, dq, quat, v, omega, gvec, state_tau)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller to calculate joint torques"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class PiConfig:
    """Pi robot configuration based on config.yaml parameters"""
    def __init__(self):
        # Configuration based on config.yaml
        self.num_actions = 20  # actions_dim: 22
        self.frame_stack = 5   # history_actor has 4 steps of history, so 5 frames total
        
        # Observation dimensions (based on obs_dict and obs_dims in config.yaml)
        self.obs_components = {
            'actions': 20,           # robot.dof_obs_size: 22
            'base_ang_vel': 3,       # Angular velocity
            'dof_pos': 20,           # Joint positions
            'dof_vel': 20,           # Joint velocities
            'projected_gravity': 3,   # Gravity projection
            # 'ref_motion_phase': 1,    # Motion phase
            'rel_fut_ref_motion_state_flat': 406  # Future reference motion state (744-335=406)
        }
        
        # History components (4 steps of history for each component)
        self.history_components = ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity']
        
        # Calculate total observation dimension (current step + history)
        current_obs_dim = sum(self.obs_components.values())
        history_obs_dim = sum([self.obs_components[comp] * 4 for comp in self.history_components])
        self.num_obs = current_obs_dim + history_obs_dim
        
        print(f"Current obs dim: {current_obs_dim}")
        print(f"History obs dim: {history_obs_dim}")
        print(f"Total obs dim: {self.num_obs}")
        
        # Control parameters (based on config.yaml)
        self.fps=200
        self.decimation = 4  # control_decimation: 4
        self.sim_duration = 260.0
        self.sim_dt = 1.0 / self.fps
        self.dt =self.decimation * self.sim_dt # Control frequency: 50Hz (200fps / 4 decimation)

        
        # Scaling factors from config.yaml
        self.obs_scales = {
            'base_ang_vel': 0.25,
            'projected_gravity': 1.0,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
            'actions': 1.0,
            'rel_fut_ref_motion_state_flat': 1.0
        }
        
        # Action scaling (from config.yaml: robot.control.action_scale)
        self.action_scale = 0.25
        
        # Action clipping limit (consistent with URCI)
        # Note: If there are still large-scale actions, try smaller values like 10.0 or 5.0
        self.clip_action_limit = 100.0  # Add action clipping limit
        
        # PD control parameters (from config.yaml: robot.control.stiffness and damping)
        # Set according to joint order and parameters in config.yaml
        self.kps = np.array([
            # Left leg (hip_pitch, hip_roll, thigh, calf, ankle_pitch, ankle_roll)
            80.0, 80.0, 80.0, 80.0, 80.0, 80.0,
            # Left arm (shoulder_pitch, shoulder_roll, upper_arm, elbow, wrist)  
            30.0, 30.0, 15.0, 30.0,
            # Right leg (hip_pitch, hip_roll, thigh, calf, ankle_pitch, ankle_roll)
            80.0, 80.0, 80.0, 80.0, 80.0, 80.0,
            # Right arm (shoulder_pitch, shoulder_roll, upper_arm, elbow, wrist)
            30.0, 30.0, 15.0, 30.0,
        ], dtype=np.double)
        
        self.kds = np.array([
            # Left leg
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
            # Left arm
            0.6, 0.6, 0.6, 0.6, 
            # Right leg
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
            # Right arm
            0.6, 0.6, 0.6, 0.6, 
        ], dtype=np.double)
        
        self.tau_limit = np.array([15.0] * 20, dtype=np.double)
        
        self.dq_limit = np.array([6.0] * 20, dtype=np.double)
        
        # Arranged according to dof_names order
        self.default_angles = np.array([
            # Left leg
            0.0,  # l_hip_pitch_joint
            0.0,  # l_hip_roll_joint
            0.0,  # l_thigh_joint
            0.0,  # l_calf_joint
            0.0,  # l_ankle_pitch_joint
            0.0,  # l_ankle_roll_joint
            # Left arm
            0.0,  # l_shoulder_pitch_joint
            0.0,  # l_shoulder_roll_joint
            0.0,  # l_upper_arm_joint
            0.0,  # l_elbow_joint
            # Right leg
            0.0,  # r_hip_pitch_joint
            0.0,  # r_hip_roll_joint
            0.0,  # r_thigh_joint
            0.0,  # r_calf_joint
            0.0,  # r_ankle_pitch_joint
            0.0,  # r_ankle_roll_joint
            # Right arm
            0.0,  # r_shoulder_pitch_joint
            0.0,  # r_shoulder_roll_joint
            0.0,  # r_upper_arm_joint
            0.0,  # r_elbow_joint
        ], dtype=np.double)
        
        # XML模型路径 (来自 config.yaml: robot.asset.xml_file)
        self.xml_path = "./description/robots/pi+_all/pi_plus_20dof_250828/xml/pi_20dof_0828.xml"
        
        # 运动长度（将在运行时设置）
        self.motion_len = -1.0  # -1 表示没有运动文件, 类型: float
        
        # Motion tracking related parameters
        self.future_ref_steps = 1  # Number of future timesteps to predict
        self.motion_data = None
        self.motion_loaded = False
        self.current_time = 0.0
        
        # JSON数据相关参数
        self.use_json_data = False
        self.json_data_array = None  # 类型: Optional[np.ndarray]
        self.json_data_loaded = False




def get_rel_fut_ref_obs_from_json(config, motion_time):
  
    if not config.use_json_data or not config.json_data_loaded:
        # If JSON data is not used, return zero vector
        print("Warning: JSON data not loaded, returning zero vector")
        return np.zeros(406, dtype=np.float32)
    
    # Calculate current timestep index
    fps = 50  # Assume data is sampled at 50fps, consistent with training
    dt = 1.0 / fps
    current_step = int(motion_time / dt)
    
    # Handle looped playback
    total_steps = config.json_data_array.shape[0]
    if total_steps == 0:
        print("Warning: JSON data is empty, returning zero vector")
        return np.zeros(406, dtype=np.float32)
    
    # Use cyclic index
    step_index = current_step % total_steps
    
    # Get data for corresponding timestep
    rel_fut_ref_data = config.json_data_array[step_index]
    
    # Ensure correct data dimensions
    if rel_fut_ref_data.shape[0] != 406:
        print(f"Warning: JSON data dimension mismatch, expected 406, got {rel_fut_ref_data.shape[0]}, truncating or padding")
        if rel_fut_ref_data.shape[0] > 406:
            rel_fut_ref_data = rel_fut_ref_data[:406]
        else:
            # Pad to 406 dimensions
            padded_data = np.zeros(406, dtype=np.float32)
            padded_data[:rel_fut_ref_data.shape[0]] = rel_fut_ref_data
            rel_fut_ref_data = padded_data
    
    return rel_fut_ref_data.astype(np.float32)

def run_sim2sim_pi_simplified(onnx_model_path, xml_path=None, motion_file_path=None, json_file_path=None):
    """Run simplified Pi robot simulation"""
    
    print(f"Starting Pi robot simulation with ONNX model: {onnx_model_path}")
    
    # Initialize configuration
    config = PiConfig()
    
    # Check if JSON data should be used
    if json_file_path and os.path.exists(json_file_path):
        print(f"JSON file detected, will use JSON data: {json_file_path}")
        try:
            json_data = load_rel_fut_ref_motion_state_flat_from_json(json_file_path)
            config.json_data_array = json_data  # type: ignore
            config.use_json_data = True
            config.json_data_loaded = True
            print(f"JSON data loaded successfully, {json_data.shape[0]} steps of data")
            
            # Calculate motion length (based on JSON data)
            fps = 30  # Assume data is 30fps
            config.motion_len = json_data.shape[0] / fps
            print(f"Motion length calculated from JSON data: {config.motion_len} seconds")
            
        except Exception as e:
            print(f"Warning: Failed to load JSON data: {e}")
            print("Will fall back to original motion data calculation method")
            config.use_json_data = False
            config.json_data_loaded = False
    
   
    
    # Update XML path
    if xml_path and os.path.exists(xml_path):
        config.xml_path = xml_path
    
    # Verify XML file exists
    if not os.path.exists(config.xml_path):
       
        raise FileNotFoundError(f"Cannot find MuJoCo XML file. Tried path: {config.xml_path}")
    
    # Load ONNX policy
    policy = ONNXPolicyWrapper(onnx_model_path)
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(config.xml_path)
    model.opt.timestep = config.sim_dt  # MuJoCo internal timestep
    # model.opt.timestep = config.dt / config.decimation  # MuJoCo internal timestep

    data = mujoco.MjData(model)
    
    print(f"MuJoCo model loaded: {config.xml_path}")
    print(f"Control frequency: {1/config.dt:.1f} Hz")
    print(f"Simulation timestep: {model.opt.timestep:.6f} s")
    print(f"Motion length: {config.motion_len} seconds")
    
    # Base position: pos: [0.0, 0.0, 0.35]
    data.qpos[2] = 0.37  # Base height
   
    # Set initial joint positions for stable standing pose
    if len(data.qpos) > 7:
        data.qpos[7:] = config.default_angles
        data.qvel[:] = 0.0  # Zero initial velocity
    
    # Initial simulation steps to stabilize physics
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Setup keyboard control
    
    # Initialize control variables
    action = np.zeros(config.num_actions, dtype=np.float32)
    
    history = {
        "actions": deque(maxlen=4),
        "base_ang_vel": deque(maxlen=4),
        "dof_pos": deque(maxlen=4),
        "dof_vel": deque(maxlen=4),
        "projected_gravity": deque(maxlen=4),
    }
    
    # Initialize history buffers
    for _ in range(4):
        history["actions"].append(np.zeros(config.num_actions, dtype=np.float32))
        history["base_ang_vel"].append(np.zeros(3, dtype=np.float32))
        history["dof_pos"].append(np.zeros(config.num_actions, dtype=np.float32))
        history["dof_vel"].append(np.zeros(config.num_actions, dtype=np.float32))
        history["projected_gravity"].append(np.zeros(3, dtype=np.float32))
    
    count_lowlevel = 0
    count_csv = 0
    
    # Motion tracking variables (consistent with URCI)
    timer = 0  # Step counter, corresponding to self.timer in URCI
    motion_time = 0.0  # Motion time, corresponding to self.motion_time in URCI
    ref_motion_phase = 0.0  # Motion phase, corresponding to self.ref_motion_phase in URCI
    
    # CSV logging setup (based on Pi robot's 22 joints)
    csv_path = f"./sim2sim_pi_simplified.csv"
    csv_headers = [
        "sim2sim_base_euler_roll", "sim2sim_base_euler_pitch", "sim2sim_base_euler_yaw",
        "sim2sim_l_hip_pitch_joint", "sim2sim_l_hip_roll_joint", "sim2sim_l_thigh_joint",
        "sim2sim_l_calf_joint", "sim2sim_l_ankle_pitch_joint", "sim2sim_l_ankle_roll_joint",
        "sim2sim_l_shoulder_pitch_joint", "sim2sim_l_shoulder_roll_joint", "sim2sim_l_upper_arm_joint",
        "sim2sim_l_elbow_joint", "sim2sim_l_wrist_joint",
        "sim2sim_r_hip_pitch_joint", "sim2sim_r_hip_roll_joint", "sim2sim_r_thigh_joint",
        "sim2sim_r_calf_joint", "sim2sim_r_ankle_pitch_joint", "sim2sim_r_ankle_roll_joint",
        "sim2sim_r_shoulder_pitch_joint", "sim2sim_r_shoulder_roll_joint", "sim2sim_r_upper_arm_joint",
        "sim2sim_r_elbow_joint", "sim2sim_r_wrist_joint"
    ]
    
    # Simulation loop
    print("Starting simulation... Press ESC to exit, use number keyboard to control")
    
    try:
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_headers)
            
            total_steps = int(config.sim_duration / config.dt)
            for step in tqdm(range(total_steps), desc="Simulation progress"):
                # 获取观测
                q, dq, quat, v, omega, gvec, state_tau = get_obs(data)
                q = q[-config.num_actions:]  # Joint positions
                dq = dq[-config.num_actions:]  # Joint velocities
                
                # Limit joint velocity to prevent instability
                dq = np.clip(dq, -config.dq_limit, config.dq_limit)
                
                # Control downsampling
                if count_lowlevel % config.decimation == 0:
                    # timer += 1
                    
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    qj_obs = (q - config.default_angles) * config.obs_scales['dof_pos']
                    dqj_obs = dq * config.obs_scales['dof_vel']
                    ang_vel_obs = omega * config.obs_scales['base_ang_vel']
                    gravity_obs = gravity_orientation * config.obs_scales['projected_gravity']
                    
                    # motion_time = timer * config.dt

                    motion_time = timer * config.dt
                    print('motion_time: ', motion_time)

                    if config.motion_len > 0:
                        ref_motion_phase = (motion_time % config.motion_len) / config.motion_len
                        print('ref_motion_phase: ', ref_motion_phase)
                        print('config.motion_len: ', config.motion_len)
                    else:
                        ref_motion_phase = 0.0
                    
                    if config.motion_len > 0 and ref_motion_phase > 1.0:
                        print(f"\nMotion ended! Phase: {ref_motion_phase:.3f}")
                      
                    
                    action_obs = action * config.obs_scales['actions']
                   
                    rel_fut_ref_obs = get_rel_fut_ref_obs_from_json(config, motion_time) * config.obs_scales['rel_fut_ref_motion_state_flat']
                
                    # actions, base_ang_vel, dof_pos, dof_vel, history_actor, projected_gravity, rel_fut_ref_motion_state_flat
                    num_actions = config.num_actions
                    timer += 1
                    
                    obs_hist = np.concatenate([
                        np.concatenate(list(history["actions"])),
                        np.concatenate(list(history["base_ang_vel"])),
                        np.concatenate(list(history["dof_pos"])),
                        np.concatenate(list(history["dof_vel"])),
                        np.concatenate(list(history["projected_gravity"])),
                    ])
                    
                    obs_buf = np.concatenate([
                        action_obs,                    # actions: (0, 20)
                        ang_vel_obs,                   # base_ang_vel: (20, 23) 
                        qj_obs,                        # dof_pos: (23, 43)
                        dqj_obs,                       # dof_vel: (43, 63)
                        obs_hist,                      # history_actor: (63, 331)
                        gravity_obs,                   # projected_gravity: (331, 334)
                        rel_fut_ref_obs               # rel_fut_ref_motion_state_flat: (335, 744)
                    ]).reshape(1, -1).astype(np.float32)  
                    
                    # 更新历史
                    history["actions"].appendleft(action_obs.copy())
                    history["base_ang_vel"].appendleft(ang_vel_obs.copy())
                    history["dof_pos"].appendleft(qj_obs.copy())
                    history["dof_vel"].appendleft(dqj_obs.copy())
                    history["projected_gravity"].appendleft(gravity_obs.copy())
                    
                    action_raw = policy(obs_buf)
                    action[:] = action_raw.flatten()
                    
                    action_clipped = np.clip(action, -config.clip_action_limit, config.clip_action_limit)
                    
                    if not np.allclose(action, action_clipped):
                        print(f"警告：动作被裁剪！原始动作范围: [{np.min(action):.3f}, {np.max(action):.3f}]")
                        print(f"裁剪后动作范围: [{np.min(action_clipped):.3f}, {np.max(action_clipped):.3f}]")
                    
                    target_q = config.default_angles + action_clipped * config.action_scale
                
                target_dq = np.zeros(config.num_actions, dtype=np.double)
                tau = pd_control(target_q, q, config.kps, target_dq, dq, config.kds)
                tau = np.clip(tau, -config.tau_limit, config.tau_limit)
                
                data.ctrl[:] = tau
                
                mujoco.mj_step(model, data)
                
                viewer.render()
                
                count_lowlevel += 1
                
                if not viewer.is_alive:
                    break
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    finally:
        viewer.close()
        print(f"Simulation completed. CSV data saved to: {csv_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simplified Pi robot simulation")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="ONNX model checkpoint path")
    parser.add_argument("--xml_path", type=str,
                       help="MuJoCo XML file path")
 
    parser.add_argument("--json_file", type=str,
                       help="JSON file path")
    
    args = parser.parse_args()
    
    # 验证ONNX模型存在
    if not os.path.exists(args.checkpoint):
        print(f"Error: ONNX model not found: {args.checkpoint}")
        sys.exit(1)
    
    # Graceful shutdown handling
    def signal_handler(sig, frame):
        print('\nGracefully shutting down...')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        run_sim2sim_pi_simplified(
            onnx_model_path=args.checkpoint,
            xml_path=args.xml_path,
            json_file_path=args.json_file
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 