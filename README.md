# Regrasp Project / 机器人重抓取项目

This repository contains perception, calibration, mesh-based cup pose estimation, and Kinova Gen3 Lite motion smoke tests for a RealSense-based regrasp pipeline.

本仓库包含基于 RealSense 的机器人重抓取流程：相机/ArUco 标定、点云清理、杯子 mesh 位姿估计、Kinova Gen3 Lite 真实运动测试，以及 Pink/Pinocchio 外部 IK 验证脚本。

## Hardware / 硬件

- Robot arm / 机械臂: Kinova Gen3 Lite
- Camera / 相机: Intel RealSense D435i
- Marker / 标定码: ArUco `DICT_6X6_250`, ID `23`, marker black border length `0.096 m`
- Gripper / 夹爪: Kinova parallel gripper
- Objects / 物体: two regular black cup meshes in `mesh/`

## Main Pipeline / 主流程

```text
RealSense RGB-D
-> ArUco dynamic calibration to Kinova base frame
-> calibrated pickup/place workspace crop
-> depth-edge and outlier filtering
-> table removal and DBSCAN clustering
-> mesh-based cup pose estimation
-> regrasp planning
-> Kortex joint-angle execution
```

```text
RealSense RGB-D
-> ArUco 动态标定到 Kinova base 坐标系
-> 使用校准好的抓取区/放置区裁剪
-> 深度边缘飞点和离群点过滤
-> 去桌面和 DBSCAN 聚类
-> 基于 mesh 的杯子位姿估计
-> 重抓取规划
-> Kortex 关节角执行
```

## Setup / 环境安装

Recommended Python / 推荐 Python:

```text
Python 3.10
```

Clone the repository / 下载仓库:

```powershell
git clone <your-github-repo-url>
cd Regrasp_Project
```

Create and activate an environment / 创建并激活环境:

```powershell
conda create -n regrasp python=3.10
conda activate regrasp
```

Install Python dependencies / 安装 Python 依赖:

```powershell
python -m pip install -r requirements.txt
```

For Pink/Pinocchio on Windows, conda-forge is often more stable / Windows 上 Pink/Pinocchio 建议优先使用 conda-forge:

```powershell
conda install -c conda-forge pink osqp qpsolvers
```

Install the Kinova Kortex API wheel / 安装 Kinova Kortex API wheel:

```powershell
python -m pip install .\kortex_api-2.7.0.post5-py3-none-any.whl
```

If the wheel is not committed, obtain it from the official Kortex package and install it locally.

如果仓库中没有提交该 wheel，请从 Kinova 官方 Kortex 包中获取并在本地安装。

## Configuration / 配置

Copy and edit robot config / 复制并修改机器人配置:

```powershell
copy .\configs\robot_config.example.yaml .\configs\robot_config.yaml
```

Local calibration files are intentionally ignored by Git because they depend on the lab setup:

以下本地标定文件依赖实验室实际布置，默认不提交到 Git:

```text
configs/robot_config.yaml
configs/table_workspace.yaml
configs/real_workspaces.yaml
configs/real_calibration.yaml
```

## Common Commands / 常用命令

View RealSense stream / 查看 RealSense 画面:

```powershell
python .\scripts\view_realsense.py
```

Calibrate pickup and place workspaces / 校准杯子散落区和放置区:

```powershell
python .\scripts\calibrate_workspaces.py
```

Clean cup point clouds / 清理杯子点云:

```powershell
python .\scripts\real_clean_cup_pointcloud_test.py --save-debug-pcd
```

Offline mesh cup pose test from a saved no-table point cloud / 用已保存 no-table 点云做离线 mesh 位姿估计:

```powershell
python .\scripts\real_mesh_cup_pose_test.py `
  --mesh big=.\mesh\bigcup.STL `
  --mesh small=.\mesh\smallcup.STL `
  --mesh-unit m `
  --mesh-axis y `
  --mesh-origin bbox_center `
  --input-pcd .\debug_pointclouds\cup_clean\<file>_pcd_base_no_table.ply `
  --input-as-scene `
  --save-debug-pcd `
  --visualize
```

Pink/Pinocchio offline IK smoke test / Pink/Pinocchio 离线 IK 测试:

```powershell
python .\scripts\test_pink_ik_gen3_lite.py --target-offset 0.02 0 0 --orientation-cost 1.0
```

Read real joints and run Pink IK without moving / 读取真机关节角并离线求解 Pink IK，不运动:

```powershell
python .\scripts\real_pink_ik_current_joints_smoke_test.py --target-offset 0.02 0 0
```

Small real movement with Pink IK and Kortex joint trajectory / Pink IK + Kortex 关节轨迹小位移真机测试:

```powershell
python .\scripts\real_pink_ik_joint_move_smoke_test.py `
  --target-offset 0.01 0 0 `
  --orientation-cost 1.0 `
  --max-joint-delta 10 `
  --duration 10
```

This script requires typing `MOVE` before execution.

该脚本执行前需要手动输入 `MOVE` 确认。

## Safety / 安全

- Real robot scripts should be run only with the emergency stop accessible.
- Keep the workspace clear before any Kortex motion script.
- Prefer joint-angle motion through `PlayJointTrajectory` or `reach_joint_angles`.
- Cartesian `reach_pose` is kept only as an experiment because it has been unreliable in this project.
- Start with small offsets, long durations, and low joint-delta limits.

- 运行真机脚本时必须保证急停可触达。
- 执行任何 Kortex 运动脚本前，先清空工作空间。
- 优先使用 `PlayJointTrajectory` 或 `reach_joint_angles` 关节角运动。
- Cartesian `reach_pose` 在本项目中不稳定，仅作为实验保留。
- 首次测试使用小位移、长 duration、低关节变化阈值。

## Repository Guide / 文件说明

See [docs/FILE_GUIDE.md](docs/FILE_GUIDE.md) for file purpose, staging recommendations, and suggested GitHub cleanup.
See [docs/SCRIPTS.md](docs/SCRIPTS.md) for detailed script usage and safety labels.

文件用途、建议提交内容和仓库整理建议见 [docs/FILE_GUIDE.md](docs/FILE_GUIDE.md)。
详细脚本用途和安全标签见 [docs/SCRIPTS.md](docs/SCRIPTS.md)。
