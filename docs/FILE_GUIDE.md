# File Guide / 文件说明

This guide explains the current repository structure, what each important file group does, and what should or should not be staged before committing to GitHub.

本文档说明当前仓库结构、主要文件用途，以及提交 GitHub 前建议 stage / 不 stage 的内容。

## Recommended Stage List / 建议提交

Stage these source, config-template, documentation, and model files:

建议提交以下源码、配置模板、文档和模型文件：

```text
README.md
requirements.txt
.gitignore
docs/
control/
perception/
utils/
scripts/
configs/robot_config.example.yaml
configs/real_calibration.yaml        # if it is a shared lab calibration / 如果这是固定实验室标定
kortex_description/
mesh/
official_kortex_examples/
aruco_6x6_id23.png
generate_aruco_marker.py
```

## Recommended Do Not Stage / 建议不要提交

These are local outputs, credentials, local calibration, or large debug files:

以下是本地输出、账号配置、本地标定或大型调试文件：

```text
configs/robot_config.yaml
configs/table_workspace.yaml
configs/real_workspaces.yaml
debug_pointclouds/
captures/
pointclouds/
sim_captures/
sim_move_outputs/
sim_pick_place_outputs/
*.whl
```

The local cup meshes in `mesh/` can be staged if collaborators need the same cup models.

如果协作者需要相同杯子模型，`mesh/` 下的 STL 可以提交。

## Directory Overview / 目录概览

### `scripts/`

Runnable scripts for calibration, perception, planning, and robot smoke tests.

可直接运行的脚本目录，用于标定、感知、规划和真机测试。

| File | Purpose / 用途 | Stage? / 是否建议提交 |
| --- | --- | --- |
| `view_realsense.py` | Shows aligned RealSense color/depth stream. / 显示 RealSense 彩色和对齐深度画面。 | Yes |
| `calibrate_table_workspace.py` | Click four table corners and save a table workspace. / 点击桌面四角，保存桌面区域。 | Yes |
| `calibrate_workspaces.py` | Click pickup and place workspace polygons. / 点击校准杯子散落区和放置区。 | Yes |
| `real_clean_cup_pointcloud_test.py` | Cleans RealSense cup clusters with edge, table, outlier, and DBSCAN filters. / 对杯子点云做边缘、桌面、离群点和聚类清理。 | Yes |
| `real_detect_cup_state_test.py` | Earlier PCA-based cup state detector. / 早期基于 PCA 的杯子状态检测。 | Yes, but legacy |
| `real_detect_cup_axis_test.py` | Detects principal axis of cup/object clusters. / 检测杯子或物体点云主轴。 | Yes |
| `real_mesh_cup_pose_test.py` | Mesh-based cup pose/model/state estimation, online or offline. / 基于 mesh 的杯子位姿、型号和状态估计，支持在线/离线。 | Yes |
| `plan_regrasp_dry_run.py` | Perception-only regrasp planning without robot motion. / 不动真机的重抓取规划 dry run。 | Yes |
| `real_regrasp_workspaces.py` | Real pickup/place cycles over calibrated workspaces. / 基于校准工作区的真实抓取放置流程。 | Yes, experimental |
| `real_regrasp_workspaces_auto.py` | Auto-confirm wrapper for real regrasp. / 全自动重抓取封装脚本。 | Yes, with caution |
| `test_pink_ik_gen3_lite.py` | Offline Pink/Pinocchio IK smoke test. / 离线 Pink/Pinocchio IK 测试。 | Yes |
| `real_pink_ik_current_joints_smoke_test.py` | Reads real joints and tests Pink IK without moving. / 读取真机关节角并离线测试 Pink IK，不运动。 | Yes |
| `real_pink_ik_joint_move_smoke_test.py` | Executes a small Pink IK joint target with Kortex `PlayJointTrajectory`. / 用 Pink IK 和 Kortex 关节轨迹执行小位移真机测试。 | Yes |
| `real_gripper_test.py` | Kinova gripper open/close/read test. / Kinova 夹爪开合和读取测试。 | Yes |
| `real_move_above_cup_test.py` | Earlier move-above-cup real robot test. / 早期移动到杯子上方真机测试。 | Yes, legacy |
| `real_pick_place_*.py` | Earlier pick-and-place scripts. / 早期 pick-and-place 脚本。 | Yes, legacy |
| `real_perception_test.py` | RealSense/ArUco perception validation. / RealSense 和 ArUco 感知验证。 | Yes |
| `real_approach_test.py` | Approach motion test. / 接近动作测试。 | Yes, experimental |
| `experimental_kortex/` | Kortex API experiments. / Kortex API 实验脚本。 | Optional |

### `perception/`

Reusable perception utilities.

可复用感知工具。

| File | Purpose / 用途 |
| --- | --- |
| `realsense_camera.py` | RealSense aligned RGB-D capture wrapper. / RealSense 对齐 RGB-D 采集封装。 |
| `aruco_detector.py` | ArUco marker detector and pose utilities. / ArUco 检测和位姿工具。 |
| `object_locator.py` | Object localization helpers. / 物体定位辅助函数。 |

### `control/`

Kortex connection and motion wrapper.

Kortex 连接和运动封装。

| File | Purpose / 用途 |
| --- | --- |
| `kortex_controller.py` | Minimal Kortex controller wrapper for joints, Cartesian trajectory tests, gripper, and joint speeds. / Kortex 控制封装，包含关节角、笛卡尔轨迹测试、夹爪和关节速度接口。 |

### `utils/`

Math helpers shared by scripts.

脚本共享数学工具。

| File | Purpose / 用途 |
| --- | --- |
| `transform_utils.py` | Homogeneous transform, quaternion, ArUco pose, and deprojection helpers. / 齐次变换、四元数、ArUco 位姿和像素反投影工具。 |

### `configs/`

Configuration files.

配置文件。

| File | Purpose / 用途 | Stage? / 是否建议提交 |
| --- | --- | --- |
| `robot_config.example.yaml` | Safe template for robot IP/account and motion safety defaults. / 机器人 IP、账号和安全参数模板。 | Yes |
| `robot_config.yaml` | Local credentials and robot connection config. / 本地账号和机器人连接配置。 | No |
| `real_calibration.yaml` | Dynamic calibration result `T_base_marker`. / 动态标定结果。 | Maybe |
| `table_workspace.yaml` | Local clicked table workspace. / 本地点击桌面区域。 | No |
| `real_workspaces.yaml` | Local pickup/place clicked workspaces. / 本地散落区/放置区。 | No |

### `mesh/`

Cup CAD meshes used by mesh-based pose estimation.

用于 mesh 位姿估计的杯子 CAD 模型。

| File | Purpose / 用途 |
| --- | --- |
| `bigcup.STL` | Large/narrow-mouth cup model. / 大杯或收口杯模型。 |
| `smallcup.STL` | Small regular cup model. / 小杯模型。 |

### `kortex_description/`

URDF, xacro, meshes, and limits for Kinova arms and grippers.

Kinova 机械臂和夹爪的 URDF、xacro、mesh 和关节限制文件。

Important file / 关键文件:

```text
kortex_description/robots/gen3_lite.urdf
```

This is used by Pink/Pinocchio scripts.

Pink/Pinocchio 脚本使用该 URDF。

### `debug_pointclouds/`, `captures/`, `pointclouds/`

Generated data and debug outputs. Do not stage by default.

生成数据和调试输出，默认不提交。

### `reference/`

Research papers, prior project code, and external references. Useful during development but not required for running the current scripts.

论文、历史项目代码和外部参考。开发时有用，但当前脚本运行不依赖它们。

For a cleaner public GitHub repository, consider moving this directory to a separate `references` branch, Git LFS, or excluding large PDFs.

如果要做更干净的公开 GitHub 仓库，可考虑把该目录移到单独分支、Git LFS，或不提交大型 PDF。

## Suggested GitHub Cleanup / 标准 GitHub 仓库整理建议

Current structure is usable, but it can be cleaner.

当前结构可以使用，但还可以更标准。

Recommended future structure:

建议后续结构：

```text
Regrasp_Project/
  README.md
  requirements.txt
  docs/
  configs/
    robot_config.example.yaml
  src/
    regrasp/
      perception/
      control/
      planning/
      geometry/
  scripts/
    calibration/
    perception/
    robot/
    ik/
  assets/
    aruco/
    mesh/
  robot_description/
    kortex_description/
  tests/
```

Do not restructure immediately before committing if the scripts are working. First commit the current working state, then reorganize in a separate cleanup commit.

如果当前脚本能运行，不建议在提交前立刻大规模移动文件。先提交当前可工作版本，再用单独 cleanup commit 整理结构。

