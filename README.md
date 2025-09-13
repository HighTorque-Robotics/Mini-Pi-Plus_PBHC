

## pi+ 的readme

### 环境安装：

``` # Assuming pwd: /Pi_mimic
conda create -n Pi_mimic python=3.8
conda activate Pi_mimic
pip install -e isaacgym/python

# Install Pi_mimic
pip install -e .
pip install -e humanoidverse/isaac_utils

```
SMPLSim需要手动下载安装 
https://github.com/ZhengyiLuo/SMPLSim
``` 
ln -s path/to/SMPLSim path/to/Pi_mimic/SMPLSim
```
```
pip install -e SMPLSim
```




### 1. 获取数据，GMR重定向数据
1. 使用GMR
https://github.com/HighTorque-Locomotion/GMR_json

在完成GMR_json的配置后，重定向获取此项目需要的动作
  
  1.1 下载lafan数据[[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) motion data] download raw LAFAN1 bvh files from [the official repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), i.e., [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip).
  1.2 
  ``` 
  # single motion
python scripts/bvh_to_robot.py --bvh_file assets/lafan/fallAndGetUp2_subject2.bvh --robot pi_football --save_path assets/pi_lafan/fallAndGetUp2_subject2_2.pkl --rate_limit
  ``` 
    


## 2. Pi_mimic使用数据
1. 
将从GMR项目获取到的pkl文件fallAndGetUp2_subject2_2.pkl保存到Pi_mimic项目下的 retargeted_motion_data/mink/pi_lafan 的文件夹下

2. 把GMR获得的数据添加脚部接触序列

``` 
python motion_source/count_pkl_contact_mask.py +input_folder=retargeted_motion_data/mink/pi_lafan +output_folder=retargeted_motion_data/mink/pi_lafan_fixed robot=pi+_20dof
```
代码是通过脚的高度和脚的速度来判断脚部是否接触地面，所以可以根据需要修改脚的高度和脚速度的阈值

3. 截取所需的帧

``` 
python robot_motion_process/motion_clip_interpolation.py --origin_file_name retargeted_motion_data/mink/pi_lafan_contact_mask/fallAndGetUp2_subject2_cont_mask_fixed.pkl --start 830 --end 1051 --end_inter_frame 25
```

4. 可视化重定向后的数据
查看截取前的数据

``` 
python robot_motion_process/vis_q_mj_pi+_20dof.py +motion_file=retargeted_motion_data/mink/pi_lafan_contact_mask/fallAndGetUp2_subject2_cont_mask_fixed.pkl
```
查看截取后的数据
```
python robot_motion_process/vis_q_mj_pi+_20dof.py +motion_file=retargeted_motion_data/mink/pi_lafan_contact_mask/fallAndGetUp2_subject2_cont_mask_fixed_inter1_E1051-25.pkl
```





## 3. 训练模型

根据训练动作不同需要修改配置文件中的参数，比如terminate_by_gravity，penalize_contacts_on等



``` 
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking_pi +terrain=terrain_locomotion_plane \
project_name=pi_dance num_envs=8192 \
+obs=motion_tracking/main_pi20dof  \
+robot=pi+_20dof/pi+_20dof \
+domain_rand=main_pi20dof \
+rewards=motion_tracking/main_pi20dof \
experiment_name=debug \
robot.motion.motion_file='retargeted_motion_data/mink/pi_lafan_contact_mask/fallAndGetUp2_subject2_cont_mask_fixed_inter1_E1051-25.pkl' \
seed=1 \
+device=cuda:0 +env=motion_tracking_pi20dof

```

## 4. 测试模型

测试使用未来帧的模型，保存未来帧数据

``` 
python humanoidverse/eval_agent_save_json.py +device=cuda:0 +env.config.enforce_randomize_motion_start_eval=False +checkpoint=logs/pi_dance/20250909_224908-debug-motion_tracking-pi_20dof/model_0.pt +robot=pi+_20dof/pi+_20dof
```



## 5. sim2sim


2. sim2sim使用未来帧的模型的模型
```
python humanoidverse/sim2sim_pi20dof.py
--checkpoint logs/new0830/pi_form2_curri/exported/model_20000.onnx
--xml_path description/robots/pi+_all/pi_plus_20dof_250828/xml/pi_20dof_0828.xml
--json_file logs/new0830/pi_form2_curri/rel_fut_ref_motion_state_flat_data.json

```

## 6 部署
把onnx转撑rknn模型

## pretrained model
查看example/pretrained_pose

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [PBHC](https://github.com/TeleHuman/PBHC#): We use `PBHC` library to build our RL codebase.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): We use `rsl_rl` library for the PPO implementation.
- [GMR](https://github.com/YanjieZe/GMR): We use `GMR` for the retargeting pipeline.

