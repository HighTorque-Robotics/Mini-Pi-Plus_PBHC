import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sympy import false
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403

# add argparse arguments
from utils.devtool import pdb_decorator
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger

import threading
# from pynput import keyboard

import pynput
from pynput import keyboard
import pdb
import numpy as np
# torch import moved after isaacgym import to avoid conflicts
import pickle
import shutil
import joblib
import json

def on_press(key, env):
    try:
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
        # Force Control
       # Force Control
        if hasattr(key, 'char'):
            if key.char == '1':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '2':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '3':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
            elif key.char == '4':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
    except AttributeError:
        pass

def listen_for_keypress(env):
    return
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()



# from humanoidverse.envs.base_task.base_task import BaseTask
# from humanoidverse.envs.base_task.omnih2o_cfg import OmniH2OCfg
@hydra.main(config_path="config", config_name="base_eval")
# @pdb_decorator
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == 'IsaacGym':
        import isaacgym
    
    # Import torch after isaacgym to avoid conflicts
    import torch
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)
    
    # Also save motion_len to the original training config file if it exists and doesn't have motion_len
    if override_config.checkpoint is not None:
        checkpoint = Path(override_config.checkpoint)
        original_config_path = checkpoint.parent / "config.yaml"
        if not original_config_path.exists():
            original_config_path = checkpoint.parent.parent / "config.yaml"
        
        if original_config_path.exists():
            # Load the original training config
            with open(original_config_path, 'r') as file:
                original_config = OmegaConf.load(file)
            
            # Check if motion_len should be added or updated
            if hasattr(config, 'obs') and hasattr(config.obs, 'motion_len'):
                if not hasattr(original_config, 'obs'):
                    original_config.obs = {}
                
                # Check if motion_len is missing or equals -1, then add/update accordingly
                original_motion_len = getattr(original_config.obs, 'motion_len', None) if hasattr(original_config, 'obs') else None
                calculated_motion_len = config.obs.motion_len
                
                config_updated = False
                
                if original_motion_len is None:
                    # motion_len doesn't exist, add it
                    original_config.obs.motion_len = calculated_motion_len
                    original_config.obs.motion_file = config.obs.motion_file
                    logger.info(f"Added motion_len ({calculated_motion_len}) to original training config")
                    config_updated = True
                elif original_motion_len == -1 and calculated_motion_len != -1:
                    # motion_len exists but is -1, add motion_len_for_deploy with real value
                    original_config.obs.motion_len_for_deploy = calculated_motion_len
                    if hasattr(config.obs, 'motion_file'):
                        original_config.obs.motion_file_for_deploy = config.obs.motion_file
                    logger.info(f"Added motion_len_for_deploy ({calculated_motion_len}) to original training config (original motion_len was -1)")
                    config_updated = True
                elif original_motion_len != -1:
                    logger.info(f"motion_len already exists in original training config: {original_motion_len}")
                else:
                    logger.info(f"Both original and calculated motion_len are -1, no update needed")
                
                if config_updated:
                    # Save the updated config back to the original file
                    with open(original_config_path, 'w') as file:
                        OmegaConf.save(original_config, file)
                    logger.info(f"Updated original training config: {original_config_path}")
            else:
                logger.warning("motion_len not found in processed config")

    # print(f"config.num_envs: {config.num_envs}"); breakpoint()
    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.num_envs = 1
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion
    env = instantiate(config.env, device=device)

    # Start a thread to listen for key press
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    EXPORT_POLICY = False
    EXPORT_ONNX = True

    checkpoint_path = str(checkpoint)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # from checkpoint path

    ROBOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exported_policy_path = os.path.join(ROBOVERSE_ROOT_DIR, checkpoint_dir, 'exported')
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split('/')[-1]
    exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

    if EXPORT_POLICY:
        export_policy_as_jit(algo.alg.actor_critic, exported_policy_path, exported_policy_name)
        logger.info('Exported policy as jit script to: ', os.path.join(exported_policy_path, exported_policy_name))
    if EXPORT_ONNX:
        example_obs_dict = algo.get_example_obs()
        export_policy_as_onnx(algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)
        logger.info(f'Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}')

    # Add data collection for plotting target_dof_pos and dof_pos
    target_dof_pos_data = []
    dof_pos_data = []
    ref_joint_pos_data = []
    actions_scaled_data = []
    
    # Add data collection for combined CSV
    rel_fut_ref_motion_state_flat_data = []
    dof_names = env.dof_names if hasattr(env, 'dof_names') else [f"joint_{i}" for i in range(env.num_dof)]
    # max_steps = 400  # Maximum number of steps to collect data
    max_steps = calculated_motion_len*50  # Maximum number of steps to collect data
    
    logger.info("Starting data collection for DOF position plotting...")
    
    # Custom evaluation loop to collect data
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    actor_state = algo._create_actor_state()
    step = 0
    algo.eval_policy = algo._get_inference_policy()
    obs_dict = env.reset_all()
    init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
    actor_state.update({"obs": obs_dict, "actions": init_actions})
    actor_state = algo._pre_eval_env_step(actor_state)
    
    try:
        while step < max_steps:
            actor_state["step"] = step
            actor_state = algo._pre_eval_env_step(actor_state)
            
            # Get actions from policy
            actions = actor_state["actions"]
            
            
            # Get current dof_pos
            current_dof_pos = env.simulator.dof_pos.clone()
            
            # Store data (only for the first environment)
            dof_pos_data.append(current_dof_pos[0].detach().cpu().numpy())
            # rel_fut_ref_motion_state_flat = env._get_obs_rel_fut_ref_motion_state_flat()
            rel_fut_ref_motion_state_flat = env.pri_rel_fut_ref_motion_state_flat[:, 3:]

            rel_fut_ref_motion_state_flat_data.append(rel_fut_ref_motion_state_flat[0].detach().cpu().numpy())
           
            
            # Step the environment
            actor_state = algo.env_step(actor_state)
            actor_state = algo._post_eval_env_step(actor_state)
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                logger.info(f"Collected data for {step}/{max_steps} steps")
                
    except KeyboardInterrupt:
        logger.info(f"Data collection interrupted at step {step}")
    
    # Convert to numpy arrays
    target_dof_pos_data = np.array(target_dof_pos_data)
    dof_pos_data = np.array(dof_pos_data)
    ref_joint_pos_data = np.array(ref_joint_pos_data)
    actions_scaled_data = np.array(actions_scaled_data)
    rel_fut_ref_motion_state_flat_data = np.array(rel_fut_ref_motion_state_flat_data)
    
    logger.info(f"Data collection completed. Collected {len(target_dof_pos_data)} samples")
    
    
    # Save rel_fut_ref_motion_state_flat to JSON
    save_rel_fut_ref_motion_state_flat_to_json(rel_fut_ref_motion_state_flat_data, checkpoint.parent)
    
    # Continue with regular evaluation if needed
    algo.evaluate_policy()





def save_rel_fut_ref_motion_state_flat_to_json(rel_fut_ref_motion_state_flat_data, save_dir):
    """
    Save rel_fut_ref_motion_state_flat data to JSON file
    
    Args:
        rel_fut_ref_motion_state_flat_data (np.ndarray): The data to save
        save_dir (Path): Directory to save the JSON file
    """
    logger.info("Saving rel_fut_ref_motion_state_flat data to JSON...")
    
    # Convert numpy array to list for JSON serialization
    data_to_save = {
        "metadata": {
            "total_steps": len(rel_fut_ref_motion_state_flat_data),
            "data_shape": list(rel_fut_ref_motion_state_flat_data.shape) if rel_fut_ref_motion_state_flat_data.size > 0 else [],
            "description": "rel_fut_ref_motion_state_flat data collected during evaluation"
        },
        "data": []
    }
    
    # Process each time step
    for step_idx, step_data in enumerate(rel_fut_ref_motion_state_flat_data):
        if step_data.size > 0:
            # Convert numpy array to list for JSON serialization
            step_entry = {
                "time_step": step_idx,
                "values": step_data.tolist()
            }
            data_to_save["data"].append(step_entry)
        else:
            # Handle empty data
            step_entry = {
                "time_step": step_idx,
                "values": None
            }
            data_to_save["data"].append(step_entry)
    
    # Save to JSON file
    json_path = save_dir / "rel_fut_ref_motion_state_flat_data.json"
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    logger.info(f"rel_fut_ref_motion_state_flat data saved to JSON: {json_path}")
    
    # Also save a simplified version with just the raw data arrays
    simplified_data = {
        "metadata": data_to_save["metadata"],
        "raw_data": rel_fut_ref_motion_state_flat_data.tolist() if rel_fut_ref_motion_state_flat_data.size > 0 else []
    }
    
    simplified_json_path = save_dir / "rel_fut_ref_motion_state_flat_data_simplified.json"
    with open(simplified_json_path, 'w') as f:
        json.dump(simplified_data, f, indent=2)
    
    logger.info(f"Simplified rel_fut_ref_motion_state_flat data saved to JSON: {simplified_json_path}")
    
    return json_path


if __name__ == "__main__":
    main()
