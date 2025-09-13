import os.path as osp
import numpy as np
import sys
import os
import pickle
import copy

sys.path.append(os.getcwd())

from utils.torch_humanoid_batch import Humanoid_Batch
import torch
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from scipy.spatial.transform import Rotation as sRot

def foot_detect(positions, thres=0.0012):
    fid_r, fid_l = 15,5 # 5: left_ankle_pitch_link    15: right_ankle_pitch_link
    positions = positions.numpy()
    velfactor, heightfactor = np.array([thres]), np.array([0.12]) 
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:,fid_l,2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.expand_dims(feet_l,axis=1)
    feet_l = np.concatenate([np.array([[1.]]),feet_l],axis=0)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:,fid_r,2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.expand_dims(feet_r,axis=1)
    feet_r = np.concatenate([np.array([[1.]]),feet_r],axis=0)
    return feet_l, feet_r

def count_pose_aa(motion):
    dof = motion['dof_pos']
    root_qua = motion['root_rot']
    # dof_new = np.concatenate((dof[:, :19], dof[:, 22:26]), axis=1)
    dof_new = np.concatenate((dof[:, :10], dof[:, 11:21]), axis=1) #删除wrist

    root_aa = sRot.from_quat(root_qua).as_rotvec()

    # Try to load dof_axis from the correct path relative to the script location
    dof_axis_path = os.path.join(os.path.dirname(__file__), '../description/robots/pi+_20dof/pi+_20dof_dof_axis_v1.npy')
    
    
    dof_axis = np.load(dof_axis_path, allow_pickle=True)
    dof_axis = dof_axis.astype(np.float32)

    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1), dof_axis * np.expand_dims(dof_new, axis=2), np.zeros((dof_new.shape[0], 3, 3))),
        axis=1).astype(np.float32)
    
    return pose_aa,dof_new

def process_motion(motion, cfg):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot)  # load forward kinematics model

    # breakpoint()

    if 'pose_aa' not in motion.keys():
        pose_aa,dof = count_pose_aa(motion=motion)
        motion['pose_aa'] = pose_aa
        motion['dof'] = dof
        pose_aa = torch.from_numpy(pose_aa).unsqueeze(0)
    else:
        pose_aa = torch.from_numpy(motion['pose_aa']).unsqueeze(0)
    root_trans = torch.from_numpy(motion['root_pos']).unsqueeze(0)

    fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)

    feet_l, feet_r = foot_detect(fk_return.global_translation_extend[0])

    motion['contact_mask'] = np.concatenate([feet_l,feet_r],axis=-1)
    motion['smpl_joints'] = fk_return.global_translation_extend[0].detach().numpy()

    return motion

def convert_format(target_data, reference_file="example/motion_data/fallAndGetUp2_subject2_cont_mask_fixed_inter0.5_E1051-25.pkl"):
    """Convert motion data format to match reference file structure"""
    print(f"Converting format using reference: {reference_file}")
    
    ref_data = joblib.load(reference_file)
    ref_key = list(ref_data.keys())[0]
    ref_structure = ref_data[ref_key]
    
    print(f"Reference file structure:")
    print(f"  Outer key: {ref_key}")
    print(f"  Fields: {list(ref_structure.keys())}")
    
    print(f"\nTarget file original structure:")
    print(f"  Fields: {list(target_data.keys())}")
    
    new_data = {}
    new_key = ref_key
    
    new_inner = {}
    
    field_mapping = {
        'root_pos': 'root_trans_offset',
        'pose_aa': 'pose_aa',
        'dof': 'dof',
        'root_rot': 'root_rot',
        'smpl_joints': 'smpl_joints',
        'fps': 'fps',
        'contact_mask': 'contact_mask'
    }
    
    for old_key, new_key_name in field_mapping.items():
        if old_key in target_data:
            value = target_data[old_key]
            
            if isinstance(value, np.ndarray):
                if old_key in ['pose_aa', 'smpl_joints']:
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                        print(f"  Converting {old_key} data type: {target_data[old_key].dtype} -> float32")
                elif old_key in ['root_pos', 'root_rot', 'dof']:
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                        print(f"  Converting {old_key} data type: {target_data[old_key].dtype} -> float32")
                elif old_key == 'contact_mask':
                    if value.dtype != np.float64:
                        value = value.astype(np.float64)
                        print(f"  Converting {old_key} data type: {target_data[old_key].dtype} -> float64")
            
            new_inner[new_key_name] = value
            print(f"  Converting field {old_key} -> {new_key_name}: {type(value).__name__}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        else:
            print(f"  Warning: missing field {old_key} in target file")
    
    missing_fields = set(ref_structure.keys()) - set(new_inner.keys())
    if missing_fields:
        print(f"  Missing fields: {missing_fields}")
        for field in missing_fields:
            if field == 'smpl_joints' and 'pose_aa' in new_inner:
                pose_shape = new_inner['pose_aa'].shape
                if len(pose_shape) == 3:
                    n_frames, n_joints, _ = pose_shape
                    new_inner['smpl_joints'] = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
                    print(f"    Creating placeholder field {field}: shape({n_frames}, {n_joints}, 3)")
    
    new_data[ref_key] = new_inner
    
    return new_data

def safe_load_pickle(file_path):
    """Safely load pickle files with multiple fallback strategies for numpy compatibility"""
    strategies = [
        # Strategy 1: Try pickle with error handling
        lambda: _try_pickle_load(file_path),
        # Strategy 2: Try joblib with error handling
        lambda: _try_joblib_load(file_path),
        # Strategy 3: Try with numpy compatibility mode
        lambda: _try_numpy_compatible_load(file_path),
    ]
    
    last_error = None
    for i, strategy in enumerate(strategies):
        try:
            print(f"Trying loading strategy {i+1}...")
            result = strategy()
            print(f"Successfully loaded with strategy {i+1}")
            return result
        except Exception as e:
            print(f"Strategy {i+1} failed: {e}")
            last_error = e
            continue
    
    raise RuntimeError(f"Failed to load {file_path} with all strategies. Last error: {last_error}")

def _try_pickle_load(file_path):
    """Try loading with standard pickle"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def _try_joblib_load(file_path):
    """Try loading with joblib"""
    return joblib.load(file_path)

def _try_numpy_compatible_load(file_path):
    """Try loading with numpy compatibility handling"""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Try to monkey patch numpy if needed
        try:
            import numpy as np
            if not hasattr(np, '_core'):
                # Create a dummy _core module if it doesn't exist
                import types
                np._core = types.ModuleType('_core')
                np._core._dtype = type(np.dtype('float32'))
                np._core._dtype_kind_to_letter = lambda x: 'f'
                np._core._dtype_from_pep3118 = lambda x: np.dtype('float32')
        except Exception:
            pass
        
        # Try pickle first
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except:
            # Fallback to joblib
            return joblib.load(file_path)

def safe_save_pickle(data, file_path):
    """Safely save data using pickle"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

@hydra.main(version_base=None, config_path="../description/robots/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    folder_path = cfg.input_folder
    if folder_path[-1]=='/':
        target_folder_path = folder_path[:-1] + '_contact_mask'
    else:
        target_folder_path = folder_path+'_contact_mask'
    os.makedirs(target_folder_path, exist_ok=True)
    target_folder_list = os.listdir(target_folder_path)
    for filename in os.listdir(folder_path):
        if filename.split('.')[0] + '_cont_mask.pkl' in target_folder_list:
            continue
        filename = filename.split('.')[0]
        motion_file = folder_path + '/' + f'{filename}.pkl'
        print(f"Processing: {motion_file}")
        
        try:
            motion_data = safe_load_pickle(motion_file)
            print(f"Loaded data type: {type(motion_data)}")
            print(f"Data keys/attributes: {dir(motion_data) if hasattr(motion_data, '__dict__') else 'No __dict__'}")
            
            # Handle different data structures
            if isinstance(motion_data, dict):
                motion_data_keys = list(motion_data.keys())
                print(f"Dictionary keys: {motion_data_keys}")
                # The motion_data itself contains the motion information, not nested under keys
                motion = process_motion(motion_data, cfg)
                
                # Apply format conversion
                save_data = convert_format(motion)
            elif hasattr(motion_data, 'keys'):
                # Handle objects with keys method
                motion_data_keys = list(motion_data.keys())
                print(f"Object keys: {motion_data_keys}")
                # The motion_data itself contains the motion information
                motion = process_motion(motion_data, cfg)
                
                # Apply format conversion
                save_data = convert_format(motion)
            else:
                # Handle direct motion data
                print(f"Direct motion data, type: {type(motion_data)}")
                motion = process_motion(motion_data, cfg)
                
                # Apply format conversion
                save_data = convert_format(motion)
            
            dumped_file = f'{target_folder_path}/{filename}_cont_mask_fixed.pkl'
            safe_save_pickle(save_data, dumped_file)
            print(f"Successfully processed: {filename}")
            print(f"Saved to: {dumped_file}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()