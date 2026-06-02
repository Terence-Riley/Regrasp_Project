# Scripts Usage Guide / 脚本使用说明

This document lists the scripts in `scripts/`, what they do, and whether they move the real robot.

本文档列出 `scripts/` 下脚本的用途，以及是否会移动真实机械臂。

## Safety Labels / 安全标签

```text
Perception only / 只做感知: does not move the robot / 不移动真机
Read only / 只读真机: connects to Kortex but sends no motion command / 连接 Kortex 但不发送运动命令
Moves robot / 会移动真机: sends Kortex motion or gripper commands / 会发送 Kortex 运动或夹爪命令
Legacy / 历史脚本: kept for reference, not the recommended current entry point / 保留参考，不是当前推荐入口
```

## Calibration and Perception / 标定与感知

### `view_realsense.py`

Perception only / 只做感知。

Shows RealSense color and aligned depth frames. Press `s` to save frames and `q`/Esc to quit.

显示 RealSense 彩色图和对齐深度图。按 `s` 保存画面，按 `q` 或 Esc 退出。

```powershell
python .\scripts\view_realsense.py
```

### `calibrate_table_workspace.py`

Perception only / 只做感知。

Click four table corners and save a table workspace polygon.

点击桌面四个角，保存桌面工作区多边形。

```powershell
python .\scripts\calibrate_table_workspace.py
```

### `calibrate_workspaces.py`

Perception only / 只做感知。

Click four corners for the pickup area, then four corners for the place area.

先点击杯子散落区四角，再点击杯子放置区四角。

```powershell
python .\scripts\calibrate_workspaces.py
```

### `real_perception_test.py`

Perception only / 只做感知。

Validates RealSense, ArUco detection, and base-frame point calculation.

验证 RealSense、ArUco 检测和 base 坐标系点计算。

### `real_clean_cup_pointcloud_test.py`

Perception only / 只做感知。

Cleans cup point clouds with depth-edge filtering, table removal, DBSCAN, and outlier filtering.

通过深度边缘过滤、去桌面、DBSCAN 和离群点过滤清理杯子点云。

```powershell
python .\scripts\real_clean_cup_pointcloud_test.py --save-debug-pcd
```

### `real_detect_cup_state_test.py`

Perception only, legacy / 只做感知，历史脚本。

Earlier cup state detector based on point-cloud PCA and cluster features.

早期基于 PCA 和 cluster 特征的杯子状态检测脚本。

### `real_detect_cup_axis_test.py`

Perception only / 只做感知。

Estimates principal axis of cup/object clusters using PCA.

使用 PCA 估计杯子或物体 cluster 的主轴。

### `real_mesh_cup_pose_test.py`

Perception only / 只做感知。

Fits `mesh/bigcup.STL` and `mesh/smallcup.STL` to observed cup clusters. Supports live RealSense and offline PLY/PCD input.

将 `mesh/bigcup.STL` 和 `mesh/smallcup.STL` 匹配到观测杯子点云。支持在线 RealSense 和离线 PLY/PCD 输入。

```powershell
python .\scripts\real_mesh_cup_pose_test.py `
  --mesh big=.\mesh\bigcup.STL `
  --mesh small=.\mesh\smallcup.STL `
  --mesh-unit m `
  --mesh-axis y `
  --mesh-origin bbox_center `
  --input-pcd .\debug_pointclouds\cup_clean\<file>_pcd_base_no_table.ply `
  --input-as-scene `
  --visualize
```

## Planning / 规划

### `plan_regrasp_dry_run.py`

Perception only / 只做感知。

Detects cup candidates in pickup workspace and generates a pick-place sequence without robot motion.

在散落区检测杯子候选，并生成不动真机的抓取/放置计划。

```powershell
python .\scripts\plan_regrasp_dry_run.py --save-debug-pcd
```

## Real Robot Tests / 真机测试

### `real_gripper_test.py`

Moves gripper / 会移动夹爪。

Tests Kinova gripper open, close, sequence, or read mode.

测试 Kinova 夹爪打开、关闭、序列动作或读取状态。

```powershell
python .\scripts\real_gripper_test.py --mode sequence
```

### `real_move_above_cup_test.py`

Moves robot / 会移动真机。

Early script that estimates a cup point and moves the tool above it using Kortex IK or Cartesian experiment modes.

早期脚本：估计杯子位置，并尝试移动到杯子上方，支持 Kortex IK 或 Cartesian 实验模式。

Use with caution. Prefer newer workspace and Pink IK tests for current development.

谨慎使用。当前开发更建议使用工作区脚本和 Pink IK 测试脚本。

### `real_approach_test.py`

Moves robot / 会移动真机。

Approach-motion test using existing Kortex motion helpers.

使用现有 Kortex 运动工具做接近动作测试。

### `real_pick_place_test.py`

Moves robot / 会移动真机。

First real pick/lift style test. Uses manually selected or detected target point and Kortex IK joint motion.

早期真实 pick/lift 测试。使用手动或检测目标点，通过 Kortex IK 和关节角运动执行。

### `real_pick_place_auto.py`

Moves robot / 会移动真机。

Automatic pick-place script using ArUco/point-cloud target estimation.

自动 pick-place 脚本，使用 ArUco/点云估计目标。

### `real_pick_place_auto_cluster.py`

Moves robot / 会移动真机。

Automatic pick-place with DBSCAN cluster detection.

使用 DBSCAN cluster 检测的自动 pick-place 脚本。

### `real_regrasp_workspaces.py`

Moves robot / 会移动真机。

Current real regrasp loop using calibrated pickup and place workspaces.

当前基于校准散落区和放置区的真实重抓取循环脚本。

```powershell
python .\scripts\real_regrasp_workspaces.py
```

### `real_regrasp_workspaces_auto.py`

Moves robot / 会移动真机。

Wrapper for fully automatic regrasp execution. Use only after manual confirmation flow is stable.

全自动重抓取封装。只有在手动确认版本稳定后再使用。

## Pink / Pinocchio IK / 外部 IK

### `test_pink_ik_gen3_lite.py`

Perception/control offline only / 离线测试。

Loads Gen3 Lite URDF and solves a small IK target using Pink/Pinocchio without connecting to Kortex.

加载 Gen3 Lite URDF，用 Pink/Pinocchio 求一个小目标 IK，不连接 Kortex。

```powershell
python .\scripts\test_pink_ik_gen3_lite.py --target-offset 0.02 0 0 --orientation-cost 1.0
```

### `real_pink_ik_current_joints_smoke_test.py`

Read only / 只读真机。

Reads real Kortex joint angles, then runs Pink IK offline from that posture. No motion command is sent.

读取真实 Kortex 关节角，再从该姿态离线运行 Pink IK。不发送运动命令。

```powershell
python .\scripts\real_pink_ik_current_joints_smoke_test.py --target-offset 0.02 0 0
```

### `real_pink_ik_joint_move_smoke_test.py`

Moves robot / 会移动真机。

Reads current joints, solves a small Pink IK target, performs safety checks, then asks for `MOVE` before executing Kortex `PlayJointTrajectory`.

读取当前关节角，用 Pink IK 求小位移目标，完成安全检查后要求输入 `MOVE`，再调用 Kortex `PlayJointTrajectory`。

```powershell
python .\scripts\real_pink_ik_joint_move_smoke_test.py `
  --target-offset 0.01 0 0 `
  --orientation-cost 1.0 `
  --max-joint-delta 10 `
  --duration 10
```

### `real_pink_absolute_position_smoke_test.py`

Read only by default; moves robot only with `--execute` / 默认只读与解 IK；只有加 `--execute` 才会移动真机。
Reads current Kortex joints, solves Pink IK for an absolute `tool_frame` target position, and can optionally execute the joint trajectory after typing `MOVEABS`.

读取当前 Kortex 关节角，对指定的绝对 `tool_frame` 目标位置求 Pink IK；如加 `--execute`，输入 `MOVEABS` 后执行 Kortex 关节轨迹。
```powershell
python .\scripts\real_pink_absolute_position_smoke_test.py `
  --target-position 0.45 0.17 0.43 `
  --target-rpy-deg 180 0 0 `
  --orientation-cost 4.0
```

### `real_pink_pick_one_cup_smoke_test.py`

Moves robot and gripper / 会移动真机和夹爪。
Detects one cup with RealSense + mesh fitting, uses the mesh center `x/y/z` as the grasp point, fixes the tool orientation to vertical-down, then executes `open -> pre-grasp -> grasp -> close -> lift`.

使用 RealSense + mesh 匹配检测一个杯子，以 mesh center 的 `x/y/z` 作为抓取点，固定工具姿态为竖直向下，然后执行 `open -> pre-grasp -> grasp -> close -> lift`。
```powershell
python .\scripts\real_pink_pick_one_cup_smoke_test.py `
  --mesh big=.\mesh\bigcup.STL `
  --mesh small=.\mesh\smallcup.STL `
  --mesh-unit m `
  --mesh-axis y `
  --mesh-origin bbox_center `
  --workspace-name pickup `
  --enable-dark-object-filter `
  --tool-rpy-deg 180 0 0
```

## Legacy Scripts / 历史脚本

### `scripts/legacy_capture/`

Early RealSense and ArUco capture scripts.

早期 RealSense 和 ArUco 采集脚本。

### `scripts/legacy_simulation/`

Early simulation, dynamic calibration, and PyBullet pick-place scripts.

早期仿真、动态标定和 PyBullet pick-place 脚本。

### `scripts/experimental_kortex/`

Kortex API experiments and smoke tests.

Kortex API 实验和 smoke test 脚本。
