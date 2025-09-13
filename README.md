

## Pi_Mimic项目使用方法介绍
下载项目：
```
git clone -b Pi_workshop_Beijing --single-branch git@github.com:HighTorque-Locomotion/Pi_mimic_workshop.git
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
pip install -e SMPLSim
```

### 2. 获取数据，GMR重定向数据
使用GMR项目进行数据集的重定向，原项目地址: https://github.com/YanjieZe/GMR
```
# create conda env 为避免环境冲突等问题，重定向在一个独立的虚拟环境中进行
conda create -n gmr python=3.10 -y
conda activate gmr

pip install -e GMR
conda install -c conda-forge libstdcxx-ng -y
```
**注：如果没有使用本项目提供的GMR而是是下载的原链接内容，注意安装前修改setup.py 中 numpy==1.24.4。**

下载pi_plus机器人配置文件: https://github.com/HighTorque-Locomotion/GMR_json

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
**注：推荐使用lafan数据集，如果使用的是AMASS数据集遇到问题，也可以通过该脚本1:1进行一次截取（格式化）。**
跌倒爬起fallAndGetUp2_subject2使用数据区间为: --start 830 --end 1051

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

根据训练动作不同需要修改配置文件中的参数，比如terminate_by_gravity，penalize_contacts_on等。\
推荐使用Nvidia RTX 4090或其他显存不小于16G的Nvidia显卡进行训练
``` 
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking_pi +terrain=terrain_locomotion_plane \
project_name=pi_dance num_envs=8192 \
+obs=motion_tracking/main_pi20dof  \
+robot=pi+_20dof/pi+_20dof \
+domain_rand=main_pi20dof \
+rewards=motion_tracking/main_pi20dof \
experiment_name=debug \
robot.motion.motion_file='retargeted_motion_data/mink/pi_lafan_contact_mask/{your_pkl_cont_mask_fixed_inter_file}.pkl' \
seed=1 \
+device=cuda:0 +env=motion_tracking_pi20dof

```

## 5. 测试模型

测试使用未来帧的模型，保存未来帧数据

``` 
python humanoidverse/eval_agent_save_json.py +device=cuda:0 +env.config.enforce_randomize_motion_start_eval=False +checkpoint={your_train_log_path}/{your_model_xxx}.pt +robot=pi+_20dof/pi+_20dof
```



## 6. sim2sim


#### sim2sim使用未来帧的模型的模型
```
python humanoidverse/sim2sim_pi20dof.py --checkpoint {your_train_log_path}/exported/{your_model_xxx}.onnx --xml_path description/robots/pi+_all/pi_plus_20dof_250828/xml/pi_20dof_0828.xml --json_file {your_train_log_path}rel_fut_ref_motion_state_flat_data.json
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
pt文件：example/pretrained_pose/model_13000.pt\
onnx文件：example/pretrained_pose/model_13000.onnx\
rknn文件：example/pretrained_pose/model_13000_Pi_fuwoup_fut_nowaist_0826.rknn\
俯卧未来帧数据：example/pretrained_pose/rel_fut_fuwo_0826.json\
仰卧未来帧数据：example/pretrained_pose/rel_fut_ref_motion_state_flat_data.json


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

