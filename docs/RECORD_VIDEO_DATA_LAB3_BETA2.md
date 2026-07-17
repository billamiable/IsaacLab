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
| CloudXR.js client | 旧流程常依赖单独解压的 cloudxr-js client | 可用官方 hosted client，也可在 `IsaacTeleop/deps/cloudxr/webxr_client/` 本地 build 6.2.0 client |
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

## 5. Pico / CloudXR.js 实机前置准备

这一节是实机 Pico 测试前的 host 侧准备。官方 Lab3 文档说明：Isaac Lab 3.0-beta2 里 `isaacteleop` 会随 `isaaclab_teleop` 自动安装，teleop 脚本启动时会自动启动 CloudXR runtime；Pico 4 Ultra / Quest 3 走 CloudXR.js WebXR client，Pico 4 Ultra 需要 HTTPS 模式。

### 5.1 网络与防火墙

先确认 Pico 和 Isaac Lab 工作站在同一个可互通网络里。不要使用禁止设备互访的访客 Wi-Fi 或企业隔离 WLAN；实机测试推荐独立 Wi-Fi 6 路由器。

宿主机查看 IP：

```bash
hostname -I
```

防火墙至少开放 CloudXR.js Web client 需要的端口：

```bash
# CloudXR WebRTC signaling
sudo ufw allow 49100/tcp

# CloudXR media stream. 官方 Lab3 web-client 文档列出 47998/udp；
# 如果现场网络/版本仍有媒体或输入问题，可临时放宽到 47998:48012/udp 对齐旧 2.3.2 经验。
sudo ufw allow 47998/udp
# sudo ufw allow 47998:48012/udp

# CloudXR built-in WSS proxy, Pico HTTPS 模式会用到
sudo ufw allow 48322/tcp

# 本地自建 CloudXR.js HTTPS dev server，默认 webpack-dev-server 端口
sudo ufw allow 8080/tcp
```

端口含义：

| 端口 | 协议 | 用途 |
| --- | --- | --- |
| `49100` | TCP | CloudXR WebRTC signaling |
| `47998` | UDP | CloudXR media stream；必要时可临时放宽 `47998:48012/udp` |
| `48322` | TCP | CloudXR WSS proxy，自签证书需要在 Pico 浏览器接受 |
| `8080` | TCP | 本地 CloudXR.js HTTPS dev server |

### 5.2 CloudXR.js client 选择

有两种方式。

方式 A：使用官方 hosted client。入口 URL 是：

```text
https://nvidia.github.io/IsaacTeleop/client/
```

当前实测时该入口会跳转到：

```text
https://nvidia.github.io/IsaacTeleop/client/v1.3.131/#/sim
```

本轮使用的 IsaacTeleop client 版本也是 `v1.3.131`。官方 hosted client 的优点是不用本地 build，也不需要宿主机开放 `8080/tcp`；缺点是不能改 client 代码，也依赖外网。

方式 B：self-host CloudXR.js client。这个更适合我们现在调 Pico motion controller、client UI、HTTPS 和缓存问题。它使用本地 `IsaacTeleop/deps/cloudxr/webxr_client/` 前端工程，依赖 NGC CloudXR.js SDK 包 `deps/cloudxr/nvidia-cloudxr-6.2.0.tgz`。

如果本地还没有 SDK tgz，先准备 CloudXR.js 6.2.0：

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacTeleop
source deps/cloudxr/.env.default
export CXR_WEB_SDK_VERSION
bash scripts/download_cloudxr_sdk.sh
test -f deps/cloudxr/nvidia-cloudxr-${CXR_WEB_SDK_VERSION}.tgz
```

也可以手动从 NGC CloudXR.js 6.2.0 下载 `nvidia-cloudxr-6.2.0.tgz`，放到 `IsaacTeleop/deps/cloudxr/` 下：

```text
https://catalog.ngc.nvidia.com/orgs/nvidia/-/resources/cloudxr-js/6.2.0
```

当前本地状态：`deps/cloudxr/nvidia-cloudxr-6.2.0.tgz`、`webxr_client/node_modules/` 和 `webxr_client/build/` 已存在。正常情况下后续 self-host 测试只需要按第 6 节“终端 A”启动 HTTPS dev server，然后 Pico 访问：

```text
https://<host-ip>:8080
```

第一次访问会看到自签证书警告，选择继续访问。这个证书只对应 `8080` 的本地 Web client 页面；CloudXR WSS proxy 的 `48322` 证书需要在连接 CloudXR 时另外接受。

### 5.3 Lab3 容器准备

从宿主机进入 `IsaacLab3/` 启动本地 overlay 容器：

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacLab3
bash docker/teleop_dev.sh start
bash docker/teleop_dev.sh enter
cd /workspace/isaaclab
```

如果遇到 extension registry/cache 权限错误，例如写 `/root/.local/share/ov/data/exts` 失败，通常是旧 root 容器留下的 volume 或 bind mount 权限不匹配。Lab3 beta2 容器按 uid/gid 1000 运行，处理方式是清理对应 named volume，或把宿主机缓存/输出目录 `chown -R 1000:1000` 后再启动。

## 6. Pico 实机 Step-by-Step 启动顺序

推荐实际测试时开三个终端。

### 终端 A：CloudXR.js Web client

如果使用官方 hosted client，这个终端不需要。

如果使用 self-host client，在宿主机运行：

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacTeleop/deps/cloudxr/webxr_client
npm run dev-server:https
```

Pico 浏览器访问：

```text
https://<host-ip>:8080
```

第一次访问时接受 `8080` 页面的自签证书。

### 终端 B：Lab3 容器

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacLab3
bash docker/teleop_dev.sh start
bash docker/teleop_dev.sh enter
cd /workspace/isaaclab
```

快速确认 teleop 包：

```bash
./isaaclab.sh -p - <<PY
import isaaclab_teleop
import isaacteleop
print("isaaclab_teleop", isaaclab_teleop.__file__)
print("isaacteleop", isaacteleop.__file__)
PY
```

### 终端 C：启动真实 Pico 录制任务

在容器内 `/workspace/isaaclab` 运行第 7 节对应任务命令。实机 Pico 建议显式加：

```text
--xr --headless --cloudxr_env cloudxrjs
```

含相机任务再加：

```text
--enable_cameras --rendering_mode balanced
```

Lab3 的区别是：`--cloudxr_env cloudxrjs` 会解析到镜像/源码内置的 CloudXR.js env profile，`--auto_launch_cloudxr` 默认开启，所以 `record_demos.py` 会在 IsaacTeleop session 启动时自动启动 CloudXR runtime 和 WSS proxy。不需要像 2.3.2 那样单独启动 runtime 容器。

### Pico 端操作

1. 戴上 Pico，打开浏览器。
2. 访问本地 client：`https://<host-ip>:8080`；或官方 client：`https://nvidia.github.io/IsaacTeleop/client/`。
3. 在 client 的 Server IP 输入 Isaac Lab 工作站 IP。
4. 如果页面提示接受 WSS proxy 证书，打开 `https://<host-ip>:48322/`，选择继续访问，看到证书接受页后回到 client。
5. 点击 Connect。
6. 连接后用 client 的 Start / Play / Reset 开始遥操作。
7. 对任务四/五，左右 Pico motion controller 分别控制左右 wrist，左右 trigger 分别控制左右 Dex1 gripper。

GUI 模式和 headless 的区别：GUI 模式下官方流程需要在 XR panel 里选择 OpenXR 并点击 Start XR；headless `record_demos.py --xr --headless` 会在 IsaacTeleop session 中自动启用 XR/CloudXR，不需要手动点 Start XR。

### 实机调试常见问题

- Pico 打不开 `https://<host-ip>:8080`：检查 `npm run dev-server:https` 是否还在运行、`8080/tcp` 是否开放、Pico 和 host 是否能互通。
- 使用官方 hosted client 时不需要 `8080/tcp`；使用 self-host client 时才需要。
- 连接时卡在证书：手动访问 `https://<host-ip>:48322/` 接受 WSS proxy 自签证书。
- 能连接但没有控制：确认任务命令有 `--xr`，并且没有传 `--teleop_device motion_controllers`。Lab3 对配置了 `env_cfg.isaac_teleop` 的任务应让脚本自动走 IsaacTeleop pipeline。
- 能操控但没有 HDF5 成功样本：`record_demos.py` 只会在 success 连续满足 `--num_success_steps` 后导出有效 demo。调 pipeline 时这是正常现象。
- 含相机任务卡顿：先用 `--rendering_mode balanced`，只录 RGB；确认不要额外打开高质量渲染或 depth observation。
- CloudXR runtime 启动时报 `Port 49100 is already in use`：通常是上一次 session 残留的 `isaacteleop.cloudxr.runtime` 占住了 signaling 端口。先检查 `ss -ltnp | grep 49100`，然后在容器里清理：

```bash
docker exec isaac-lab-base-300b2 bash -lc '
pkill -f "isaacteleop.cloudxr.runtime" || true
rm -f /root/.cloudxr/run/ipc_cloudxr
'
```
- XR 启动时报 `failed to find gpu foundation devices`：优先检查 `docker exec isaac-lab-base-300b2 nvidia-smi`。如果容器内出现 `Failed to initialize NVML: Unknown Error`，不是任务代码问题，重启 Lab3 容器：

```bash
cd /home/yujie/workspace/yujie/iProject/customer/VeOV/pico/from_yanzi/cloudxr-runtime-blueprint/INTERNAL_examples/isaac-lab-teleop/IsaacLab3
bash docker/teleop_dev.sh stop
bash docker/teleop_dev.sh start
```

## 7. 真实 Pico 录制命令

真实 Pico 录制统一使用 `record_demos.py`。对配置了 `env_cfg.isaac_teleop` 的任务，**不要传** `--teleop_device motion_controllers`。如果传了，Lab3 脚本会强制走 legacy `teleop_devices` 路径，反而绕开新的 IsaacTeleop pipeline。

默认情况下：

- `--cloudxr_env cloudxrjs` 适合 Quest/Pico WebXR client。
- `--auto_launch_cloudxr` 已是默认值；脚本会自动启动 CloudXR runtime。
- 没有 Pico / 不想启动 CloudXR 时，才加 `--cloudxr_env none --no-auto_launch_cloudxr`。
- 含相机任务加 `--enable_cameras`，推荐 `--rendering_mode balanced`。
- 成功样本落盘仍受 `EXPORT_SUCCEEDED_ONLY` 控制；如果任务 success 没连续满足，能操控但 HDF5 可能没有有效 demo。

### 任务一：GR1T2 PickPlace

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --device cuda:0 --rendering_mode balanced \
  --xr --headless --cloudxr_env cloudxrjs \
  --dataset_file /workspace/host/out/isaaclab3/task1_gr1t2_pickplace.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务二：Galbot Visuomotor Stack Cube

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0 \
  --device cuda:0 --enable_cameras --rendering_mode balanced \
  --xr --headless --cloudxr_env cloudxrjs \
  --dataset_file /workspace/host/out/isaaclab3/task2_galbot_stack_cube_visuomotor.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务三：Franka IK Stack Cube

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
  --device cuda:0 --rendering_mode balanced \
  --xr --headless --cloudxr_env cloudxrjs \
  --dataset_file /workspace/host/out/isaaclab3/task3_franka_stack_cube_ik_abs.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务四：G1 Dex1 FixedBase StackCube Reachability

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0 \
  --device cuda:0 --rendering_mode balanced \
  --xr --headless --cloudxr_env cloudxrjs \
  --dataset_file /workspace/host/out/isaaclab3/task4_g1_dex1_stack_cube_reachability.hdf5 \
  --num_demos 0 --num_success_steps 10
```

### 任务五：G1 Dex1 FixedBase StackCube Visuomotor

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0 \
  --device cuda:0 --enable_cameras --rendering_mode balanced \
  --xr --headless --cloudxr_env cloudxrjs \
  --dataset_file /workspace/host/out/isaaclab3/task5_g1_dex1_stack_cube_visuomotor.hdf5 \
  --num_demos 0 --num_success_steps 10
```

当前实机状态：该命令已通过 Pico + 官方 hosted client 跑通，也已通过 Pico + self-host client 跑通。第一阶段验收目标是连通性、遥操作输入、三路相机和 HDF5 数据链路跑通；已知可优化项是 motion controller 和 gripper/wrist 的空间跟随仍有偏差，后续应放到 retargeter/calibration 调整里处理。

录制完成后转视频：

```bash
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
  --input_file /workspace/host/out/isaaclab3/task5_g1_dex1_stack_cube_visuomotor.hdf5 \
  --output_dir /workspace/host/out/isaaclab3/task5_g1_dex1_stack_cube_visuomotor_videos \
  --input_keys ego_cam left_wrist_cam right_wrist_cam \
  --framerate 30
```

## 8. 当前 G1 Dex1 visuomotor 实现说明

任务五新增内容：

- `g1_dex1_stack_cube_visuomotor_env_cfg.py`：在 G1 Dex1 stack-cube 基础上增加三路 RGB camera observation。
- `g1_dex1_visuomotor.usda` overlay：相机挂在已有机器人 USD 层级下，CameraCfg 使用 `spawn=None` 绑定到 overlay 中的 camera prim。
- `record_g1_dex1_mock_pico_visuomotor_demo.py`：headless mock Pico 录制脚本，生成 HDF5 和 summary JSON。

相机数据只启用 `rgb`。之前 2.3.2/早期测试里出现过 headless 下 `DistanceToImagePlaneSD` 或 `LdrColorSD` empty buffer 的问题；当前任务五先关闭 depth，只保留 RGB，以降低 Lab3 headless SyntheticData 出错面。

## 9. 后续可选项

- 如果需要无 Pico 对任务一/二/三也做“动作语义级” mock 轨迹，应按任务各自 action layout 补专用 mock script；当前它们在 Lab3 registry 和真实 Pico 入口上已可用。
- 任务五真实录制后可继续调三路相机 FOV、相机 offset、cube 初始位置、success termination 稳定时间。
- 如果要把 mock HDF5 也作为训练数据，需要明确标记 `mock/*` 字段，避免和真实 Pico 数据混用。
