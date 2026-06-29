# Isaac Lab 3.0-beta2 Teleop / Visuomotor 验证流程

本文档是 `IsaacLab/docs/RECORD_VIDEO_DATA.md` 的 Isaac Lab 3.0-beta2 迁移版。目标是覆盖原来的五个 task，并说明 Lab3 与 Isaac Lab 2.3.2 在 Pico / CloudXR / IsaacTeleop 启动方式上的差异。

参考官方文档：<https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/how-to/cloudxr_teleoperation.html>

## 1. Lab3 与 2.3.2 的关键差异

| 项 | Isaac Lab 2.3.2 | Isaac Lab 3.0-beta2 |
| --- | --- | --- |
| 容器形态 | 本地维护的 teleop 镜像，额外挂 CloudXR Runtime / IsaacLab 代码 | 官方 Lab3 容器内已包含 Isaac Lab + IsaacTeleop 依赖，本项目只做本地 overlay |
| 代码覆盖 | 常用整仓挂载到 `/workspace/isaaclab` | 推荐只 overlay `source/`、`scripts/`、`docs/`、`tools/`，避免覆盖镜像内 `_isaac_sim` 等运行状态 |
| Teleop 包 | 2.3.2 需要项目内维护 teleop patch | 3.0-beta2 里 `isaaclab_teleop` 和 `isaacteleop` 已在镜像/源码布局中可用 |
| `record_demos.py` | 常显式传 `--teleop_device motion_controllers` | 对配置了 `env_cfg.isaac_teleop` 的任务，不要传 `--teleop_device`；脚本会自动走 IsaacTeleop 栈 |
| CloudXR 启动 | 依赖旧版外部 runtime/服务 | `record_demos.py` 默认 `--cloudxr_env cloudxrjs --auto_launch_cloudxr`，可自动启动 runtime |
| GUI Start XR | GUI 里需要手动点 Start XR/AR | headless 录制不需要 GUI Start XR；脚本创建 IsaacTeleop device 时启动 XR/CloudXR session |
| 输出目录 | 常用 `/workspace/host/out/...` | 继续使用 `/workspace/host/out/...`，但建议命名到 `out/isaaclab3/...` 下避免和 2.3.2 混淆 |

注意：Lab3 的 `--headless` 仍可用，但官方已逐步把“默认 headless / 显式 `--viz`”作为方向。当前本地脚本仍保留 `--headless`，便于和旧流程对齐。

## 2. 启动 Lab3 本地 overlay 容器

从宿主机进入 `IsaacLab3/`：

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacLab3
bash docker/teleop_dev.sh start
```

默认容器和输出挂载：

```text
container: isaac-lab-base-300b2
repo:      IsaacLab3/{source,scripts,docs,tools} -> /workspace/isaaclab/{source,scripts,docs,tools}
out:       ../out -> /workspace/host/out
```

进入容器：

```bash
bash docker/teleop_dev.sh enter
cd /workspace/isaaclab
```

停止容器：

```bash
bash docker/teleop_dev.sh stop
```

快速确认 teleop 包可见：

```bash
./isaaclab.sh -p - <<'PY'
import isaaclab_teleop
import isaacteleop
print("isaaclab_teleop", isaaclab_teleop.__file__)
print("isaacteleop", isaacteleop.__file__)
PY
```

## 3. 当前支持的任务

| 任务 | Lab3 状态 | 说明 |
| --- | --- | --- |
| 任务一 `Isaac-PickPlace-GR1T2-Abs-v0` | 上游已有 | GR1T2 IsaacTeleop 任务，适合真实 Pico 直接录制 |
| 任务二 `Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0` | 上游已有 | 含相机观测，真实 Pico 录制时加 `--enable_cameras` |
| 任务三 `Isaac-Stack-Cube-Franka-IK-Abs-v0` | 上游已有 | Franka motion controller 参考任务 |
| 任务四 `Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0` | 本项目注册 | G1 + Dex1 低维 reachability / stack-cube 任务 |
| 任务五 `Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0` | 本项目注册 | G1 + Dex1 + 三路 RGB 相机；本轮已做 headless mock HDF5 录制验证 |

Registry 检查：

```bash
./isaaclab.sh -p - <<'PY'
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
for task_id in [
    "Isaac-PickPlace-GR1T2-Abs-v0",
    "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-IK-Abs-v0",
    "Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0",
    "Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0",
]:
    print(task_id, task_id in gym.registry)
PY
```

本次验证结果：五个 task 均为 `True`。

## 4. 无 Pico 的 mock 验证

### 4.1 官方固定 G1 Pink IK mock smoke

这个实验用于验证 Lab3 的 motion-controller-like 输入可以进入真实 Pink IK action，并驱动官方固定上半身 G1 env。输出在 `out/isaaclab3/mock_pico_g1_fixed_base_ik/`。

```bash
./isaaclab.sh -p scripts/tools/render_mock_pico_g1_fixed_base_ik_video.py \
  --headless --device cuda:0 --rendering_mode balanced \
  --frames 144 --fps 24 --width 960 --height 540 \
  --out-mp4 /workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik/mock_pico_g1_fixed_base_ik.mp4 \
  --out-json /workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik/mock_pico_g1_fixed_base_ik_summary.json \
  --out-frames /workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik/frames
```

通过标准：右腕有明显运动，Pink IK 跟踪误差可接受，视频非空。

### 4.2 G1 Dex1 低维 reachability / 物理抓取 mock smoke

这个实验使用任务四的动作路径：右手 mock controller 位姿控制右腕，右 trigger 控制右 Dex1 gripper。默认是物理抓取模式，不再用 attach 吸附。

```bash
./isaaclab.sh -p scripts/tools/render_mock_pico_g1_dex1_stack_cube_video.py \
  --headless --device cuda:0 --rendering_mode balanced \
  --grasp-mode physical \
  --frames 192 --fps 24 --width 960 --height 540 \
  --out-mp4 /workspace/host/out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/mock_pico_g1_dex1_stack_cube_physical_grasp_v2.mp4 \
  --out-json /workspace/host/out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/mock_pico_g1_dex1_stack_cube_physical_grasp_v2.json \
  --out-frames /workspace/host/out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/frames
```

已验证结果：`passed: true`，`used_assisted_grasp: false`，`ever_assisted_attached: false`。

### 4.3 G1 Dex1 visuomotor 三路相机 HDF5 mock smoke

这个实验验证任务五：同一条 G1 Dex1 action 路径下，同时记录低维状态和三路 RGB：

- `ego_cam`
- `left_wrist_cam`
- `right_wrist_cam`

```bash
./isaaclab.sh -p scripts/tools/record_g1_dex1_mock_pico_visuomotor_demo.py \
  --device cuda:0 --rendering_mode balanced \
  --frames 96 --warmup-steps 16 --side right --cube cube_2 \
  --dataset-file /workspace/host/out/isaaclab3/g1_dex1_stack_cube/visuomotor_mock/g1_dex1_mock_pico_visuomotor_demo.hdf5 \
  --summary /workspace/host/out/isaaclab3/g1_dex1_stack_cube/visuomotor_mock/g1_dex1_mock_pico_visuomotor_demo.json
```

将 HDF5 中三路相机转为 MP4：

```bash
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
  --input_file /workspace/host/out/isaaclab3/g1_dex1_stack_cube/visuomotor_mock/g1_dex1_mock_pico_visuomotor_demo.hdf5 \
  --output_dir /workspace/host/out/isaaclab3/g1_dex1_stack_cube/visuomotor_mock/videos \
  --input_keys ego_cam left_wrist_cam right_wrist_cam \
  --framerate 24 --demo_id 0
```

本次已验证输出：

```text
summary: passed true
actions: shape 18
HDF5: data/demo_0, num_samples 96, success true
ego_cam:         (96, 256, 256, 3), uint8
left_wrist_cam:  (96, 256, 256, 3), uint8
right_wrist_cam: (96, 256, 256, 3), uint8
videos: demo_0_ego_cam.mp4, demo_0_left_wrist_cam.mp4, demo_0_right_wrist_cam.mp4
```

## 5. 真实 Pico 录制命令

真实 Pico 录制统一使用 `record_demos.py`。对配置了 `env_cfg.isaac_teleop` 的任务，**不要传** `--teleop_device motion_controllers`。如果传了，Lab3 脚本会强制走 legacy `teleop_devices` 路径，反而绕开新的 IsaacTeleop pipeline。

默认情况下：

- `--cloudxr_env cloudxrjs` 已是默认值；适合 Quest/Pico WebXR client。
- `--auto_launch_cloudxr` 已是默认值；脚本会自动启动 CloudXR runtime。
- 没有 Pico / 不想启动 CloudXR 时，才加 `--cloudxr_env none --no-auto_launch_cloudxr`。
- 含相机任务加 `--enable_cameras`，推荐 `--rendering_mode balanced`。
- 成功样本落盘仍受 `EXPORT_SUCCEEDED_ONLY` 控制；如果任务 success 没连续满足，能操控但 HDF5 可能没有有效 demo。

### 任务一：GR1T2 PickPlace

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --device cuda:0 --rendering_mode balanced \
  --dataset_file /workspace/host/out/isaaclab3/task1_gr1t2_pickplace.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务二：Galbot Visuomotor Stack Cube

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0 \
  --device cuda:0 --enable_cameras --rendering_mode balanced \
  --dataset_file /workspace/host/out/isaaclab3/task2_galbot_stack_cube_visuomotor.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务三：Franka IK Stack Cube

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
  --device cuda:0 --rendering_mode balanced \
  --dataset_file /workspace/host/out/isaaclab3/task3_franka_stack_cube_ik_abs.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务四：G1 Dex1 FixedBase StackCube Reachability

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0 \
  --device cuda:0 --rendering_mode balanced \
  --dataset_file /workspace/host/out/isaaclab3/task4_g1_dex1_stack_cube_reachability.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务五：G1 Dex1 FixedBase StackCube Visuomotor

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0 \
  --device cuda:0 --enable_cameras --rendering_mode balanced \
  --dataset_file /workspace/host/out/isaaclab3/task5_g1_dex1_stack_cube_visuomotor.hdf5 \
  --num_demos 0 --num_success_steps 10
```

录制完成后转视频：

```bash
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
  --input_file /workspace/host/out/isaaclab3/task5_g1_dex1_stack_cube_visuomotor.hdf5 \
  --output_dir /workspace/host/out/isaaclab3/task5_g1_dex1_stack_cube_visuomotor_videos \
  --input_keys ego_cam left_wrist_cam right_wrist_cam \
  --framerate 30
```

## 6. Pico 实机连接步骤

1. 启动 Lab3 容器：

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacLab3
bash docker/teleop_dev.sh start
bash docker/teleop_dev.sh enter
cd /workspace/isaaclab
```

2. 在容器内运行对应任务的 `record_demos.py` 命令。Lab3 会通过 `--cloudxr_env cloudxrjs` 默认路径启动 CloudXR runtime。

3. 在 Pico 端打开对应 WebXR / CloudXR client 页面并连接服务器。GUI 模式文档里的 Start XR/AR 是 GUI session 入口；headless `record_demos.py` 不需要手动点这个按钮。

4. Pico 连接后按客户端 START / reset 交互开始遥操作。对任务四/五，左右 motion controller 分别映射到左右腕部，左右 trigger 分别控制左右 Dex1 gripper。

5. 达成 success 并保持 `--num_success_steps` 后，demo 才写入 HDF5。若只是调通实时控制但没有完成 success，文件可能没有成功 episode，这是预期行为。

## 7. 当前 G1 Dex1 visuomotor 实现说明

任务五新增内容：

- `g1_dex1_stack_cube_visuomotor_env_cfg.py`：在 G1 Dex1 stack-cube 基础上增加三路 RGB camera observation。
- `g1_dex1_visuomotor.usda` overlay：相机挂在已有机器人 USD 层级下，CameraCfg 使用 `spawn=None` 绑定到 overlay 中的 camera prim。
- `record_g1_dex1_mock_pico_visuomotor_demo.py`：headless mock Pico 录制脚本，生成 HDF5 和 summary JSON。

相机数据只启用 `rgb`。之前 2.3.2/早期测试里出现过 headless 下 `DistanceToImagePlaneSD` 或 `LdrColorSD` empty buffer 的问题；当前任务五先关闭 depth，只保留 RGB，以降低 Lab3 headless SyntheticData 出错面。

## 8. 后续可选项

- 如果需要无 Pico 对任务一/二/三也做“动作语义级” mock 轨迹，应按任务各自 action layout 补专用 mock script；当前它们在 Lab3 registry 和真实 Pico 入口上已可用。
- 任务五真实录制后可继续调三路相机 FOV、相机 offset、cube 初始位置、success termination 稳定时间。
- 如果要把 mock HDF5 也作为训练数据，需要明确标记 `mock/*` 字段，避免和真实 Pico 数据混用。
