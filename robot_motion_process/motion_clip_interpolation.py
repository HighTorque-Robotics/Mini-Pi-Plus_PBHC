import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import joblib
import argparse
from pathlib import Path
DOF_AXIS_FILE = "./description/robots/pi+_20dof/pi+_20dof_dof_axis_v1.npy"

def lower_dof_interpolation(start_dof,end_dof,nframe):
    """
    Interpolation for lower body DOFs with knee modification
    For pi_20dof: left leg (0-5), right leg (10-15)
    """
    mid_frame = nframe // 2
    quat_frame = mid_frame // 2
    
    # Extract left and right leg DOFs (pi_20dof order)
    left_lower_dof = np.linspace(start_dof[:6], end_dof[:6],
                    num=mid_frame+1,
                    endpoint=False)[1:].reshape(-1, 6)
    right_lower_dof = np.linspace(start_dof[10:16], end_dof[10:16],
                    num=mid_frame+1,
                    endpoint=False)[1:].reshape(-1, 6)
    
    # Knee modification (index 3 for left leg, index 13 for right leg in original array)
    left_knee_1 = np.linspace(start_dof[3], 1.0,
                    num=quat_frame).reshape(-1, 1)
    left_knee_2 = np.linspace(1.0, end_dof[3],
                    num=quat_frame+1).reshape(-1, 1)
    right_knee_1 = np.linspace(start_dof[13], 1.0,
                    num=quat_frame).reshape(-1, 1)
    right_knee_2 = np.linspace(1.0, end_dof[13],
                    num=quat_frame+1).reshape(-1, 1)
    
    left_lower_dof[:,3] = np.concatenate((left_knee_1, left_knee_2), axis=0).squeeze(-1)
    right_lower_dof[:,3] = np.concatenate((right_knee_1, right_knee_2), axis=0).squeeze(-1)

    left_lower_dof = np.concatenate((left_lower_dof, np.tile(left_lower_dof[-1], (nframe-left_lower_dof.shape[0], 1))), axis=0)
    right_lower_dof = np.concatenate((np.tile(right_lower_dof[0], (nframe-right_lower_dof.shape[0], 1)),right_lower_dof), axis=0)
    
    lower_dof = np.concatenate((left_lower_dof,right_lower_dof), axis=1)
    return lower_dof

def correct_rot(root_rot,root_trans,dof):
    origin_root_rot = root_rot[0]
    q_correction = R.from_quat(origin_root_rot).inv()
    correct_root_rot = (q_correction * R.from_quat(root_rot)).as_quat()
    correct_root_trans = root_trans @ (q_correction.as_matrix().T)
    
    euler_initial = R.from_quat(origin_root_rot).as_euler('xyz',degrees=False)
    theta = euler_initial[1]

    print('correct theta: ',theta)
    
    # For pi_20dof: left hip pitch (0) and right hip pitch (10)
    dof[:,[0,10]] += theta

    return correct_root_trans,correct_root_rot,dof


def convert_pkl(data, output_filename,contact_mask):
    data = data.astype(np.float32)
    root_trans = data[:, :3]
    root_qua = data[:, 3:7]
    dof_new = data[:, 7:]
    root_aa = R.from_quat(root_qua).as_rotvec()

    dof_axis = np.load(DOF_AXIS_FILE, allow_pickle=True)
    dof_axis = dof_axis.astype(np.float32)

    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1), dof_axis * np.expand_dims(dof_new, axis=2), np.zeros((data.shape[0], 3, 3))),
        axis=1).astype(np.float32)

    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof_new,
        "root_rot": root_qua,
        "smpl_joints": np.zeros_like(pose_aa),
        "fps": 30,
        "contact_mask":contact_mask
    }

    print("Output DOF Shape",dof_new.shape)
    print("Output Filename: ", output_filename+".pkl")
    all_data = {}
    all_data[output_filename] = data_dump
    joblib.dump(all_data, f'{output_filename}.pkl')


def interpolate_motion(input_data, start_ext_frames, end_ext_frames, default_start_pose, default_end_pose, output_pkl,contact_mask,fix_root_rot, knee_modify):
    root_trans = input_data[:, :3]
    root_rot = input_data[:, 3:7]
    dof_pos = input_data[:, 7:]

    # root_trans,root_rot,dof_pos = correct_rot(root_rot,root_trans,dof_pos)

    start_rot_aa = R.from_quat(root_rot[0]).as_euler('ZYX')
    end_rot_aa = R.from_quat(root_rot[-1]).as_euler('ZYX')
    start_rot_fixed_z = start_rot_aa[0]
    end_rot_fixed_z = end_rot_aa[0]

    default_start_rt = default_start_pose[0:3]
    default_start_rr = default_start_pose[3:7]
    default_start_dof = default_start_pose[7:]
    default_start_rr_aa = R.from_quat(default_start_rr).as_euler('ZYX')
    
    default_end_rt = default_end_pose[0:3]
    default_end_rr = default_end_pose[3:7]
    default_end_dof = default_end_pose[7:]
    default_end_rr_aa = R.from_quat(default_end_rr).as_euler('ZYX')

    # 注释掉起始处插值逻辑，只保留end插值
    # 起始处插值
    # start_rr, start_dof = [], []
    # if start_ext_frames > 0:
    #     # root trans
    #     start_z = np.linspace(default_start_rt[2],
    #                           root_trans[0, 2],
    #                           start_ext_frames)
    #     start_root_trans = np.zeros((start_ext_frames, 3))
    #     start_root_trans[:, 0] = root_trans[0, 0]  # X保持首帧值
    #     start_root_trans[:, 1] = root_trans[0, 1]  # Y保持首帧值
    #     start_root_trans[:, 2] = start_z

    #      # dof pos
    #     # start_dof = np.linspace(default_start_dof, dof_pos[0],
    #     #                         num=start_ext_frames + 1,
    #     #                         endpoint=False)[1:].reshape(-1, 20)
    #     if knee_modify:
    #         # For pi_20dof: process legs with special knee interpolation, arms with linear interpolation
    #         # Extract arm DOFs: left_arm (6-9) + right_arm (16-19) = 8 DOFs total
    #         left_arm_start = np.linspace(default_start_dof[6:10], dof_pos[0][6:10],
    #                                 num=start_ext_frames + 1,
    #                                 endpoint=False)[1:].reshape(-1, 4)
    #         right_arm_start = np.linspace(default_start_dof[16:20], dof_pos[0][16:20],
    #                                 num=start_ext_frames + 1,
    #                                 endpoint=False)[1:].reshape(-1, 4)
    #         # Process legs (indices 0-5 and 10-15)
    #         lower_start_dof = lower_dof_interpolation(default_start_dof[np.r_[0:6, 10:16]], 
    #                                                  dof_pos[0][np.r_[0:6, 10:16]], 
    #                                                  start_ext_frames)
    #         # Reconstruct in pi_20dof order: left_leg + left_arm + right_leg + right_arm
    #         start_dof = np.concatenate((lower_start_dof[:,:6],    # left leg
    #                                    left_arm_start,           # left arm  
    #                                    lower_start_dof[:,6:],    # right leg
    #                                    right_arm_start), axis=1) # right arm
    #     else: 
    #         #  linear interpolation
    #         start_dof = np.linspace(default_start_dof, dof_pos[0],
    #                                 num=start_ext_frames + 1,
    #                                 endpoint=False)[1:].reshape(-1, 20)

    #     # root rot
    #     if not fix_root_rot:
    #         # breakpoint()
    #         rotations = R.from_euler('ZYX', 
    #                             [   np.concatenate((start_rot_aa[0:1],default_start_rr_aa[1:])), 
    #                                 np.concatenate((start_rot_aa[0:1],start_rot_aa[1:]))  ])
    #         # breakpoint()
    #         times = np.linspace(0, 1, start_ext_frames)
    #         # slerp 插值
    #         slerp = Slerp([0, 1], rotations)
    #         # interp_rots = slerp(times).as_quat()
    #         interp_rots = slerp(times).as_euler('ZYX')
    #         # start_rr = np.array([np.array([start_rot_fixed_z,q[1],q[2]]) for q in interp_rots])
    #         start_rr = R.from_euler('ZYX',interp_rots).as_quat()

    # 结束处插值
    end_rr, end_dof = [], []
    if end_ext_frames > 0:
        # 根位移处理（仅Z轴插值）
        end_z = np.linspace(root_trans[-1, 2],
                            default_end_rt[2],
                            end_ext_frames)
        end_root_trans = np.zeros((end_ext_frames, 3))
        end_root_trans[:, 0] = root_trans[-1, 0]  # X保持首帧值
        end_root_trans[:, 1] = root_trans[-1, 1]  # Y保持首帧值
        end_root_trans[:, 2] = end_z

        # dof pos
        # end_dof = np.linspace(dof_pos[-1], default_end_dof,
        #                       num=end_ext_frames + 1)[1:].reshape(-1, 20)
        if knee_modify:
            # For pi_20dof: process legs with special knee interpolation, arms with linear interpolation
            # Extract arm DOFs: left_arm (6-9) + right_arm (16-19) = 8 DOFs total
            left_arm_end = np.linspace(dof_pos[-1][6:10], default_end_dof[6:10],
                                    num=end_ext_frames + 1)[1:].reshape(-1, 4)
            right_arm_end = np.linspace(dof_pos[-1][16:20], default_end_dof[16:20],
                                    num=end_ext_frames + 1)[1:].reshape(-1, 4)
            # Process legs (indices 0-5 and 10-15)
            lower_end_dof = lower_dof_interpolation(dof_pos[-1][np.r_[0:6, 10:16]], 
                                                   default_end_dof[np.r_[0:6, 10:16]], 
                                                   end_ext_frames)
            # Reconstruct in pi_20dof order: left_leg + left_arm + right_leg + right_arm
            end_dof = np.concatenate((lower_end_dof[:,:6],    # left leg
                                     left_arm_end,           # left arm
                                     lower_end_dof[:,6:],    # right leg
                                     right_arm_end), axis=1) # right arm
        else:
            end_dof = np.linspace(dof_pos[-1], default_end_dof,num=end_ext_frames + 1)[1:].reshape(-1, 20)
        
        if not fix_root_rot:
            # root rot
            # end_rotations = R.from_quat([root_rot[-1], default_end_rr])
            end_rotations = R.from_euler('ZYX', 
                                        [   np.concatenate((end_rot_aa[0:1],default_end_rr_aa[1:])), 
                                            np.concatenate((end_rot_aa[0:1],end_rot_aa[1:]))  ])
            # times = np.linspace(0, 1, end_ext_frames + 2)[1:-1]
            times = np.linspace(1,0,end_ext_frames)
            slerp = Slerp([0, 1], end_rotations)
            # interp_rots = slerp(times).as_quat()
            interp_rots = slerp(times).as_euler('ZYX')
            # end_rr = np.array([q for q in interp_rots])
            # end_rr = np.array([np.array([end_rot_fixed_z,q[1],q[2]]) for q in interp_rots])
            end_rr = R.from_euler('ZYX',interp_rots).as_quat()

    # 修改：只保留原始数据和end插值，不添加start插值
    new_root_trans = np.vstack([
        root_trans,  # 只保留原始数据，不添加start插值
        end_root_trans
    ])

    if not fix_root_rot:
        new_root_rot = np.vstack([
            root_rot,  # 只保留原始数据，不添加start插值
            end_rr
        ])
    else:
        total_frame = input_data.shape[0] + end_ext_frames  # 移除start_ext_frames
        # Use start pose for the whole sequence when root rotation is fixed
        new_root_rot = np.tile(default_start_rr, (total_frame, 1))

    new_dof_pos = np.vstack([
        dof_pos,  # 只保留原始数据，不添加start插值
        end_dof
    ])

    # # root trans
    # first_vector = root_trans[0, :].reshape(1, 3)
    # last_vector = root_trans[-1, :].reshape(1, 3)
    # arr_start = np.tile(first_vector, (30, 1))
    # arr_end = np.tile(last_vector, (30, 1))
    # new_root_trans = np.vstack((arr_start, root_trans, arr_end))

    # first_vector = root_rot[0, :].reshape(1, 4)
    # last_vector = root_rot[-1, :].reshape(1, 4)
    # arr_start = np.tile(first_vector, (30, 1))
    # arr_end = np.tile(last_vector, (30, 1))
    # new_root_rot = np.vstack((arr_start, root_rot, arr_end))

    output_data = np.concatenate((new_root_trans, new_root_rot, new_dof_pos), axis=1)

    # np.save('test.npy', output_data)
    convert_pkl(output_data, output_pkl,contact_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_file_name', type=str, help="Origin File name, shape (nframe, 27)",
                        default='dance1_subject2.pkl')
    parser.add_argument('--fix_root_rot', type=bool, help="Fix default pose root rot. A DEBUG option, not a regular usage.",
                        default=False)
    parser.add_argument('--knee_modify', type=bool, help="Modify knee dof when motion interpolation",
                        default=False)
    parser.add_argument('--start', type=int, help="Start frame", default=0)
    parser.add_argument('--end', type=int, help="End frame", default=-1)

    parser.add_argument('--default_pose', type=str, help="Default Pose File name, shape (27,). Used for both start and end if start/end poses not specified.")
    parser.add_argument('--default_start_pose', type=str, help="Default Start Pose File name, shape (27,)")
    parser.add_argument('--default_end_pose', type=str, help="Default End Pose File name, shape (27,)")
    parser.add_argument('--start_inter_frame', type=int, help="Start Inter frame", default=30)
    parser.add_argument('--end_inter_frame', type=int, help="End Inter frame", default=30)
    args = parser.parse_args()

    origin_file_name = args.origin_file_name
    data = next(iter(joblib.load(f'{args.origin_file_name}').values()))
    
    print("Input DOF Shape: ",data['dof'].shape)
    if args.end ==-1:
        args.end = data['dof'].shape[0]

    print(f"Clip to {(args.start,args.end)}")
    dof = data['dof'][args.start:args.end]
    root_trans = data['root_trans_offset'][args.start:args.end]
    root_rot = data['root_rot'][args.start:args.end]
    contact_mask = data['contact_mask'][args.start:args.end]

    fix_root_rot = args.fix_root_rot
    knee_modify = args.knee_modify
    
    input_data = np.concatenate((root_trans,root_rot,dof), axis=1)

    # Handle default poses
    if args.default_start_pose is not None:
        default_start_pose = np.load(args.default_start_pose)
    elif args.default_pose is not None:
        default_start_pose = np.load(args.default_pose)
    else:
        # Default pose for pi_20dof (27 parameters: 3 trans + 4 rot + 20 dof)
        # Joint order: left_leg(6) + left_arm(4) + right_leg(6) + right_arm(4)
        # ###正
        default_start_pose = np.array([-8.1646895e-01, -2.2623995e-01,  2.0000000e-01, 
                                    #   0.0, 1.0, 0.0, 1.0,  #伏卧
                                    #   0.0, -1.0, 0.0, 1.0,  #仰卧
                                     6.0086644e-01,-3.7277272e-01,  6.0086644e-01,  3.7277272e-01, #伏卧
                                    #   0.0479, 0.7054, -0.0479, 0.7054, #仰卧初始帧的状态
                                      -0., 0.0, 0.0, 0., -0., 0.0,  # left leg (0-5)
                                      0., -0, 0.0, 0,              # left arm (6-9)
                                      -0., 0.0, 0.0, 0., -0., 0.0,  # right leg (10-15)
                                      0., 0, 0.0, 0])            # right arm (16-19)
        ## 向右转
        # default_start_pose = np.array([0.0, 0.0, 0.375, 
        #                               0.0, 0.0, 0.7071, 0.7071,  # 90-degree rotation around z-axis
        #                               -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg (0-5)
        #                               0., 0, 0.0, -1.46,              # left arm (6-9)
        #                               -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg (10-15)
        #                               0., 0, 0.0, -1.46])            # right arm (16-19)

    if args.default_end_pose is not None:
        default_end_pose = np.load(args.default_end_pose)
    elif args.default_pose is not None:
        default_end_pose = np.load(args.default_pose)
    else:
        # Use the same default pose as start pose if not specified
        default_end_pose = np.array([0.0, 0.0, 0.375, 
                                      0.0, 0.0, 0.7071, 0.7071,  # 90-degree rotation around z-axis
                                      -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg (0-5)
                                      0., 0, 0.0, 0,              # left arm (6-9)
                                      -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg (10-15)
                                      0.,0, 0.0, 0])            # right arm (16-19)

    start_ext_frames = args.start_inter_frame
    end_ext_frames = args.end_inter_frame

    if False:
        cit = [1.,1.]
    else:
        cit = [1,1]
    print("Contact in interpolation: ",cit)

    if knee_modify:
        # 由于只对end插值，start部分的contact_mask为空
        # if start_ext_frames%2==0:
        #     contact_mask_start = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (start_ext_frames//2-1, 1)),np.tile([1.,0.], (start_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)
            
        # else:
        #     contact_mask_start = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (start_ext_frames//2-1, 1)),np.array([[1.,1.]]),np.tile([1.,0.], (start_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)

        if end_ext_frames%2==0:
            contact_mask_end = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (end_ext_frames//2-1, 1)),np.tile([1.,0.], (end_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)
        else:
            contact_mask_end = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (end_ext_frames//2-1, 1)),np.array([[1.,1.]]),np.tile([1.,0.], (end_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)
    else:
        # 由于只对end插值，start部分的contact_mask为空
        # contact_mask_start = np.tile([cit], (start_ext_frames, 1))
        contact_mask_end = np.tile([cit], (end_ext_frames, 1))
    # contact_mask_start = np.tile([cit], (start_ext_frames, 1))
    # contact_mask_end = np.tile([cit], (end_ext_frames, 1))

    # 修改：只连接原始contact_mask和end插值的contact_mask
    contact_mask = np.concatenate((contact_mask, contact_mask_end), axis=0)

    output_filename = str(Path(args.origin_file_name).parent / Path(args.origin_file_name).stem) + f'_inter{cit[0]}_E{args.end}-{end_ext_frames}'


    interpolate_motion(
        input_data=input_data,
        start_ext_frames=start_ext_frames,
        end_ext_frames=end_ext_frames,
        default_start_pose=default_start_pose,
        default_end_pose=default_end_pose,
        output_pkl=output_filename,
        contact_mask=contact_mask,
        fix_root_rot=fix_root_rot,
        knee_modify = knee_modify
    )