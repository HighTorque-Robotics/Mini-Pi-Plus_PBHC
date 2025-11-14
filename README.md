# Mini Pi Plus Fall and Get Up Full Process Implementation
![1730344376083](Video/logo2.jpg)
![Video Cover Description](https://github.com/HighTorque-Robotics/Pi-Plus_PBHC/raw/main/Video/ab60337d9e5ab015f1fa9f51d9106051.gif)

### Download the Project:
```bash
git clone https://github.com/HighTorque-Robotics/Mini-Pi-Plus_PBHC
```

### 1. Environment Installation:

```bash
# Assuming pwd: /Pi_mimic_workshop-main
conda create -n Pi_mimic python=3.8
conda activate Pi_mimic
pip install -e {your_path_to_issacgym}isaacgym/python

# Install Pi_mimic
pip install -e .
pip install -e humanoidverse/isaac_utils
```

SMPLSim Installation. Original project address:
https://github.com/ZhengyiLuo/SMPLSim

```bash
git clone https://github.com/ZhengyiLuo/smplx.git
pip install -e SMPLSim
```

### 2. Obtain Data & GMR Retargeting Data
Use the GMR project for dataset retargeting. Original project address: https://github.com/YanjieZe/GMR

```bash
# Create a Conda environment. To avoid environment conflicts, retargeting is performed in an independent virtual environment
conda deactivate
conda create -n gmr python=3.10 -y
conda activate gmr

pip install -e GMR
conda install -c conda-forge libstdcxx-ng -y
```

**Note: If you are not using the GMR provided in this project but downloaded the content from the original link, modify `numpy==1.24.4` in setup.py before installation.**

**Note: To facilitate your use of HighTorque Robotics robots, the following modifications are already included in this project. If you need to add your own robot files, you can refer to the following content.**

#### Add Files to Corresponding Directories:

1. Add the JSON file to the `GMR/general_motion_retargeting/ik_configs/` folder.
2. Add the robot's `pi_plus_24dof_250826` file to the `/GMR/assets/` folder.

#### Modify the `general_motion_retargeting/params.py` File:
1. Add the XML path to `ROBOT_XML_DICT`:
```python
ROBOT_XML_DICT = {
  # Existing configurations
  "pi_football": ASSET_ROOT / "pi_plus_24dof_250826" / "xml" / "pi_22dof_0826.xml",
}
```

2. Add the path of the JSON file provided by this project to `IK_CONFIG_DICT`:
```python
IK_CONFIG_DICT = {
  # Existing configurations

  "bvh": {
    # Existing configurations

    "pi_football": IK_CONFIG_ROOT / "bvh_to_pi_football.json"
  }
}
```

3. Add other robot configurations:
```python
ROBOT_BASE_DICT = {
  # Existing configurations

  "pi_football": "base_link"
}

VIEWER_CAM_DISTANCE_DICT = {
  # Existing configurations

  "pi_football": 2.0
}
```

#### Download the LaFan Dataset & Retarget:
For user convenience, this project already includes the complete LaFan1 dataset. Official dataset address: https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip.

```bash
# Single motion retargeting. To avoid environment conflicts, retargeting is performed in the gmr virtual environment
Note: It is recommended to use the LaFan dataset. The fall and get up motion uses fallAndGetUp2_subject2. If you encounter issues with the AMASS dataset, you can also use this script for 1:1 clipping (formatting).
conda activate gmr

python GMR/scripts/bvh_to_robot.py --bvh_file GMR/MotionData/lafan1/{your_bvh_file}.bvh --robot pi_football --save_path GMR/RetargetData/lafan1/pi_football/{your_bvh_file}.pkl --rate_limit
```

## 3. Pi_mimic Data Preprocessing
Save the PKL files obtained from the GMR project to the `retargeted_motion_data/mink/pi_lafan` directory under the Pi_mimic project.

#### Add Foot Contact Sequences

```bash
# Switch from the gmr environment to the Pi_mimic environment for the following steps
conda activate Pi_mimic 

python motion_source/count_pkl_contact_mask.py +input_folder=retargeted_motion_data/mink/pi_plus_lafan +output_folder=retargeted_motion_data/mink/pi_plus_lafan_contact_mask robot=pi+_20dof
```

The code determines whether the feet are in contact with the ground based on foot height and foot velocity. You can modify the thresholds for foot height and foot velocity as needed (code modification location: `motion_source/count_pkl_contact_mask.py`, line 74).

#### Clip Required Frames

```bash
python robot_motion_process/motion_clip_interpolation.py --origin_file_name retargeted_motion_data/mink/pi_plus_lafan_contact_mask/{your_pkl_cont_mask_fixed_file}.pkl --start {start num} --end {end num} --end_inter_frame 25
```

**Data range for fall and get up (fallAndGetUp2_subject2): --start 1183 --end 1372**

#### Visualize Retargeted Data
View data before clipping:

```bash
python robot_motion_process/vis_q_mj_pi+_20dof.py +motion_file=retargeted_motion_data/mink/pi_plus_lafan_contact_mask/{your_pkl_cont_mask_fixed_file}.pkl
```

View data after clipping:
```bash
python robot_motion_process/vis_q_mj_pi+_20dof.py +motion_file=retargeted_motion_data/mink/pi_plus_lafan_contact_mask/{your_pkl_cont_mask_fixed_inter_file}.pkl
```

## 4. Model Training

Modify parameters in the configuration file according to different training motions. For example, the training parameters for fall and get up have disabled termination due to gravity倾倒. Set `terminate_by_gravity` to `False` in the `motion_tracking_pi20dof.yaml` file under `Mini-Pi-Plus_PBHC/humanoidverse/config/env`, and leave the joint penalty term empty (`penalize_contacts_on[]`) in the `pi+_20dof.yaml` file under `Mini-Pi-Plus_PBHC/humanoidverse/config/robot/pi+_20dof`.

It is recommended to use an Nvidia RTX 4090 or other Nvidia graphics cards with at least 16GB of VRAM for training.

```bash
python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking_pi +terrain=terrain_locomotion_plane project_name=pi_dance num_envs=8192 +obs=motion_tracking/main_pi20dof +robot=pi+_20dof/pi+_20dof +domain_rand=main_pi20dof +rewards=motion_tracking/main_pi20dof experiment_name=debug robot.motion.motion_file='retargeted_motion_data/mink/pi_lafan_contact_mask/{your_pkl_cont_mask_fixed_inter_file}.pkl' seed=1 +device=cuda:0 +env=motion_tracking_pi20dof
```

## 5. Model Testing

Test with the future-frame model and save future-frame data:

```bash
python humanoidverse/eval_agent_save_json.py +device=cuda:0 +env.config.enforce_randomize_motion_start_eval=False +checkpoint={your_train_log_path}/{your_model_xxx}.pt +robot=pi+_20dof/pi+_20dof
```

## 6. Sim2Sim

#### Sim2Sim with Future-Frame Model
```bash
python humanoidverse/sim2sim_pi20dof.py --checkpoint {your_train_log_path}/exported/{your_model_xxx}.onnx --xml_path description/robots/pi+_all/pi_plus_20dof_250828/xml/pi_20dof_0828.xml --json_file {your_train_log_path}/rel_fut_ref_motion_state_flat_data.json
```

## 7. Sim2Real
Convert ONNX to RKNN model.

To avoid environment conflicts, create an independent environment for RKNN conversion:
```bash
conda create -n rknn_model python=3.8
conda activate rknn_model

pip install rknn-toolkit2
pip install --upgrade pillow
```

Modify the model loading and saving paths in the script file:
```python
# Modify loading path
print("--> Loading model")
ret = rknn.load_onnx("{your_path_to_load}/your_policy.onnx")

# Modify output path
OUT_DIR = "{your_path_to_save}"
RKNN_MODEL_PATH = "{}/policy_from_onnx.rknn".format(OUT_DIR)
```

Run the conversion script:
```bash
python onnx2rknn.py
```

### Pretrained Models Provided by This Project
- PT file: `example/pretrained_pose/model_12000.pt`
- ONNX file: `example/pretrained_pose/exported/model_12000.onnx`
- RKNN file: `example/pretrained_pose/model_12000.rknn`
- Supine future-frame data: `example/pretrained_pose/rel_fut_yangwo_0911.json`

#### Policy File Transfer
Remotely transfer to the robot via SCP:
```bash
cd {your_rknn_path}
scp {your_rknn_file}.rknn hightorque@{your_robot_IP}:~/sim2real_master/src/sim2real/policy/up/
scp {your_json_file}.json hightorque@{your_robot_IP}:~/sim2real_master/src/sim2real/future/up/
```

#### Modify Sim2Real Master Configuration File on the Robot
Modify `{your_demo_yaml_file}.yaml`:

```yaml
# In the fuwo.yaml file under /home/hightorque/sim2real_master/src/sim2real/config/up
# Replace the loaded model with {your_rknn_file}
policy_name: "up/{your_rknn_file}.rknn"

# Replace the future-frame data with {your_json_file}
future_file_name: "up/{your_json_file}.json"
```

```yaml
# In the yangwo.yaml file under /home/hightorque/sim2real_master/src/sim2real/config/up
# Replace the loaded model with {your_rknn_file}
policy_name: "up/{your_rknn_file}.rknn"

# Replace the future-frame data with {your_json_file}
future_file_name: "up/{your_json_file}.json"
```

#### Start Running
```bash
cd sim2real_master
catkin build            					                # Compile
source devel/setup.bash 					                # Refresh environment variables
roslaunch sim2real_master joy_control_pi_plus.launch		# Start joystick control node
```

#### Joystick Operation
```
# Switch the joystick to DEFAULT mode
LT + RT + START                      # Stand up
# Place the robot in a supine or prone position on the ground
LT + RT + LB                         # Robot automatically stands up
# Then enter walking state
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [PBHC](https://github.com/TeleHuman/PBHC#): We use the `PBHC` library to build our RL codebase.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): We use the `rsl_rl` library for the PPO implementation.
- [GMR](https://github.com/YanjieZe/GMR): We use `GMR` for the retargeting pipeline.




# Mini Pi Plus跌倒爬起全流程实现（中文翻译）                           
  ![1730344376083](Video/logo2.jpg)
![视频封面描述](https://github.com/HighTorque-Robotics/Pi-Plus_PBHC/raw/main/Video/ab60337d9e5ab015f1fa9f51d9106051.gif)
### 下载项目：
```
git clone https://github.com/HighTorque-Robotics/Mini-Pi-Plus_PBHC
```
### 1. 环境安装：

``` # Assuming pwd: /Pi_mimic_workshop-main
conda create -n Pi_mimic python=3.8
conda activate Pi_mimic
pip install -e {your_path_to_issacgym}isaacgym/python

# Install Pi_mimic
pip install -e .
pip install -e humanoidverse/isaac_utils
```
SMPLSim安装，原项目地址: 
https://github.com/ZhengyiLuo/SMPLSim

```
git clone https://github.com/ZhengyiLuo/smplx.git
pip install -e SMPLSim
```

### 2. 获取数据，GMR重定向数据
使用GMR项目进行数据集的重定向，原项目地址: https://github.com/YanjieZe/GMR
```
# create conda env 为避免环境冲突等问题，重定向在一个独立的虚拟环境中进行
conda deactivate
conda create -n gmr python=3.10 -y
conda activate gmr

pip install -e GMR
conda install -c conda-forge libstdcxx-ng -y
```
**注：如果没有使用本项目提供的GMR而是是下载的原链接内容，注意安装前修改setup.py 中 numpy==1.24.4。**

**注：为了便于您使用高擎机电的机器人，本项目中已包含以下修改。若另需添加自己的机器人文件也可参考以下内容添加。**

#### 添加文件到相应目录下：

1.添加json文件到GMR/general_motion_retargeting/ik_configs/文件夹下

2.添加机器人的pi_plus_24dof_250826文件到/GMR/assets/文件夹下

#### 修改general_motion_retargeting/params.py文件：
1. ROBOT_XML_DICT 添加xml路径：
```
ROBOT_XML_DICT = {
  #原有配置
  "pi_football": ASSET_ROOT / "pi_plus_24dof_250826" /"xml"/ "pi_22dof_0826.xml",
}
```
2. IK_CONFIG_DICT 添加本项目提供的json文件路径：
```
IK_CONFIG_DICT = {
  #原有配置

  "bvh":{
    #原有配置

    "pi_football": IK_CONFIG_ROOT / "bvh_to_pi_football.json"
    }
}
```
3. 添加其他机器人配置：
```
ROBOT_BASE_DICT = {
  #原有配置

  "pi_football": "base_link"
}

VIEWER_CAM_DISTANCE_DICT = {
  #原有配置

  "pi_football": 2.0
}
```

#### 下载lafan数据集与重定向: 
为方便用户使用，本项目已包含完整lafan1数据集，数据集官方地址：https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip.

```
# 单个动作重定向 为避免环境冲突等问题，重定向在gmr虚拟环境中进行
注：推荐使用lafan数据集，跌倒爬起使用的是fallAndGetUp2_subject2；如果使用的是AMASS数据集遇到问题，也可以通过该脚本1:1进行一次截取（格式化）。
conda activate gmr

python GMR/scripts/bvh_to_robot.py --bvh_file GMR/MotionData/lafan1/{your_bvh_file}.bvh --robot pi_football --save_path GMR/RetargetData/lafan1/pi_football/{your_bvh_file}.pkl --rate_limit
```


## 3. Pi_mimic 数据预处理
将从GMR项目获取到的pkl文件保存至Pi_mimic项目下的 retargeted_motion_data/mink/pi_lafan 目录

#### 添加脚部接触序列

``` 
# 以下步骤操作均从gmr环境切换到Pi_mimic环境中
conda activate Pi_mimic 

python motion_source/count_pkl_contact_mask.py +input_folder=retargeted_motion_data/mink/pi_plus_lafan +output_folder=retargeted_motion_data/mink/pi_plus_lafan_contact_mask robot=pi+_20dof
```
代码是通过脚的高度和脚的速度来判断脚部是否接触地面，所以可以根据需要修改脚的高度和脚速度的阈值（代码修改位置motion_source/count_pkl_contact_mask.py ，74 line）

#### 截取所需的帧

``` 
python robot_motion_process/motion_clip_interpolation.py --origin_file_name retargeted_motion_data/mink/pi_plus_lafan_contact_mask/{your_pkl_cont_mask_fixed_file}.pkl --start {start num} --end {end num} --end_inter_frame 25
```
**  跌倒爬起fallAndGetUp2_subject2使用数据区间为: --start 1183 --end 1372 **

#### 可视化重定向数据
查看截取前的数据

``` 
python robot_motion_process/vis_q_mj_pi+_20dof.py +motion_file=retargeted_motion_data/mink/pi_plus_lafan_contact_mask/{your_pkl_cont_mask_fixed_file}.pkl
```
查看截取后的数据
```
python robot_motion_process/vis_q_mj_pi+_20dof.py +motion_file=retargeted_motion_data/mink/pi_plus_lafan_contact_mask/{your_pkl_cont_mask_fixed_inter_file}.pkl
```



## 4. 训练模型

根据训练动作不同需要修改配置文件中的参数，例如如跌到爬起的训练参数已把因重力倾倒终止关闭，Mini-Pi-Plus_PBHC/humanoidverse/config/env路径下的motion_tracking_pi20dof.yaml文件中terminate_by_gravity设置为False，Mini-Pi-Plus_PBHC/humanoidverse/config/robot/pi+_20dof路径下的pi+_20dof.yaml文件中惩罚关节项置空penalize_contacts_on[]。\
推荐使用Nvidia RTX 4090或其他显存不小于16G的Nvidia显卡进行训练
```
 python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking_pi +terrain=terrain_locomotion_plane project_name=pi_dance num_envs=8192 +obs=motion_tracking/main_pi20dof  +robot=pi+_20dof/pi+_20dof +domain_rand=main_pi20dof +rewards=motion_tracking/main_pi20dof experiment_name=debug robot.motion.motion_file='retargeted_motion_data/mink/pi_lafan_contact_mask/{your_pkl_cont_mask_fixed_inter_file}.pkl' seed=1 +device=cuda:0 +env=motion_tracking_pi20dof
```

## 5. 测试模型

测试使用未来帧的模型，保存未来帧数据

``` 
python humanoidverse/eval_agent_save_json.py +device=cuda:0 +env.config.enforce_randomize_motion_start_eval=False +checkpoint={your_train_log_path}/{your_model_xxx}.pt +robot=pi+_20dof/pi+_20dof
```



## 6. sim2sim


#### sim2sim使用未来帧的模型的模型
```
python humanoidverse/sim2sim_pi20dof.py --checkpoint {your_train_log_path}/exported/{your_model_xxx}.onnx --xml_path description/robots/pi+_all/pi_plus_20dof_250828/xml/pi_20dof_0828.xml --json_file {your_train_log_path}/rel_fut_ref_motion_state_flat_data.json
```

## 7. sim2real
把onnx转换为rknn模型，
为了避免环境冲突，单独创建rknn转换环境：
```
conda create -n rknn_model  python=3.8
conda activate rknn_model

pip install rknn-toolkit2
pip install --upgrade pillow
```
修改脚本文件中加载、保存文件路径：
```
# 修改加载路径
print("--> Loading model")
    ret = rknn.load_onnx("{your_path_to_load}/your_policy.onnx")

# 修改输出路径
OUT_DIR = "{your_path_to_save}"
    RKNN_MODEL_PATH = "{}/policy_from_onnx.rknn".format(OUT_DIR)
```
运行转换脚本：
```
python onnx2rknn.py
```

### 本项目提供预训练模型
pt文件：example/pretrained_pose/model_12000.pt
onnx文件：example/pretrained_pose/exported/model_12000.onnx
rknn文件：example/pretrained_pose/model_12000.rknn
仰卧未来帧数据：example/pretrained_pose/rel_fut_yangwo_0911.json


#### policy文件传输
通过scp远程传输至机器人
```
cd {your_rknn_path}
scp {your_rknn_file}.rknn hightorque@{your_robot_IP}:~/sim2real_master/src/sim2real/policy/up/
scp {your_json_file}.json hightorque@{your_robot_IP}:~/sim2real_master/src/sim2real/future/up/
```
#### 修改机器人上sim2real_master配置文件
修改{your_demo_yaml_file}.yaml

```
在/home/hightorque/sim2real_master/src/sim2real/config/up下fuwo.yaml文件的
# 替换加载模型为{your_rknn_file}
policy_name:"up/{your_rknn_file}.rknn"

# 替换未来帧数据为{your_rknn_file}
future_file_name:"up/{your_json_file}.json"
```

```
在/home/hightorque/sim2real_master/src/sim2real/config/up下yangwo.yaml文件的
# 替换加载模型为{your_rknn_file}
policy_name:"up/{your_rknn_file}.rknn"

# 替换未来帧数据为{your_rknn_file}
future_file_name:"up/{your_json_file}.json"
```
#### 启动运行
```
cd sim2real_master
catkin build            					                #编译
source devel/setup.bash 					                #刷新环境变量
roslaunch sim2real_master joy_control_pi_plus.launch		#启动手柄控制节点
```
#### 手柄操作
```
#使用手柄切换至DEFAULT模式下
LT+RT+START                      #站立
#将机器人仰卧或者俯卧在地上
LT+RT+LB                         #机器人自动站立
#随后进入行走状态
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [PBHC](https://github.com/TeleHuman/PBHC#): We use `PBHC` library to build our RL codebase.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): We use `rsl_rl` library for the PPO implementation.
- [GMR](https://github.com/YanjieZe/GMR): We use `GMR` for the retargeting pipeline.

