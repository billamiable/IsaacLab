# Isaac Lab Teleop - 录制遥操作数据（完整流程）

## 版本信息

| 组件 | 版本 |
|------|------|
| Isaac Lab | 2.3.2 |
| CloudXR Runtime | 6.0.1-webrtc |
| CloudXR JS Client | 6.0.2-beta-pid |
| Node.js | v24.12.0 |
| npm | v11.6.2 |
| Dockerfile | `Dockerfile.2.3.2` |
| 镜像 Tag | `isaac-lab-teleop:2.3.2` |
| 文档内任务 | 任务一：`Isaac-PickPlace-GR1T2-Abs-v0`；任务二：`Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0`；任务三：`Isaac-Stack-Cube-Franka-IK-Abs-v0`（motion controllers） |
| 遥操设备 | handtracking / **`motion_controllers`**（右手柄位姿 + 扳机夹爪；详见任务三）(Pico 4 Ultra) |

---

## 1. 服务端环境准备

### 1.1 安装 Node.js

Pico 4 Ultra 的 WebXR 浏览器要求 HTTPS 连接，需要在服务端运行 cloudxr-js 的 dev-server。先确认 Node.js 已安装：

```bash
npm -v    # 期望: 11.6.2
node -v   # 期望: v24.12.0
```

如未安装，参考 [Node.js 官方安装指南](https://nodejs.org/) 安装对应版本。

### 1.2 安装 cloudxr-js 客户端依赖

进入 cloudxr-js 解压目录，安装依赖。

### 1.3 配置 WSS 代理 (HAProxy)

Pico 4 Ultra 的浏览器强制要求 HTTPS，因此需要一个 WSS 代理将 `wss://` 请求转发到 CloudXR Runtime 的 `ws://49100` 端口。

推荐使用 cloudxr-js 文档中的 HAProxy Docker 方案，参考：

> `cloudxr-js-early-access_6.0.2-beta-pid/release/docs/documents/Networking_Setup.html`
> 章节: *Example 1: Development proxy (Docker / HAProxy)*

---

## 2. 构建并启动 Isaac Lab 容器

### 2.1 构建镜像

```bash
cd INTERNAL_examples/isaac-lab-teleop/single-container
docker build -f Dockerfile.2.3.2 -t isaac-lab-teleop:2.3.2 .
```

### 2.2 宿主机：开放 X11（启动容器前执行）

容器内 Isaac Sim 使用宿主机的 `DISPLAY`，需允许本机上的连接（含 root / Docker）。**在执行下一小节的 `docker run` 之前**，在宿主机终端执行：

```bash
xhost +local:
```

用完后若希望收紧权限，可在宿主机执行 `xhost -local:`（按需）。是否使用 `xhost` 请结合本机安全策略；若未开放，可能出现无法打开窗口、`X11 connection refused` 等问题。

### 2.3 启动容器与 [yujie-dev](https://github.com/billamiable/IsaacLab/tree/yujie-dev) 代码

镜像自带的是上游 Isaac Lab，**任务二**等需要本分支补丁时，任选下面一种方式与容器对齐：

- **方式 A：bind mount 本机已 clone 的仓库（下文 `docker run` 默认写法）**  
  - **适用**：开发、频繁改 `yujie-dev`；希望保存后容器里**立刻**用到新代码；**任务二**直接依赖本机 policy/任务配置。  
  - **做法**：`docker run` 保留 **`-v <your_isaaclab_repo_path>:/workspace/isaaclab`**。宿主机目录须为含 `isaaclab.sh`、`source/` 的仓库根（可先 `git clone`，见下）。  
  - **额外一步（方式 A 专有，每个新容器做一次）**：挂载会**盖住**镜像里原来的 `/workspace/isaaclab`，纯 Git 克隆通常**没有** `_isaac_sim`。必须在容器内执行 **`ln -sfn /isaac-sim _isaac_sim`**（见下文命令），否则 `isaaclab.sh` 无法指向容器内的 Isaac Sim。

- **方式 B：不挂载，启动后 `docker cp` 拷贝补丁文件**  
  - **适用**：不想把本机仓库路径挂进容器（例如交付/固定环境）；或只改**少量文件**即可。  
  - **做法**：`docker run` 里**删掉** **`-v <your_isaaclab_repo_path>:/workspace/isaaclab \`** 整行，使用镜像内置 `/workspace/isaaclab`；容器运行后在**宿主机**把本机仓库中的文件 **`docker cp` 到容器内同相对路径**（见本节末尾示例）。**任务二**须把 `yujie-dev` 里相关改动都拷进去。  
  - **一般无需 `ln`**：沿用镜像里原工作区，通常已有可用的 `_isaac_sim`。

**`<your_isaaclab_repo_path>`**：本机 Isaac Lab 根目录的**绝对路径**，须与方式 A 中 `docker run -v` **左侧**一致。

若尚无该目录，在**宿主机**克隆（最后一项为目标路径，**父目录须已存在**）：

```bash
git clone -b yujie-dev --single-branch https://github.com/billamiable/IsaacLab.git <your_isaaclab_repo_path>
```

已有仓库则 `cd` 到目录后 `git checkout yujie-dev && git pull`。

容器默认 CMD 为 `sleep infinity`，启动后 CloudXR 会自动运行，不会自动跑遥操作脚本。

**`docker run`（方式 A：含 mount；方式 B 请删除带 `<your_isaaclab_repo_path>` 的那一行）**：

```bash
docker run --rm -it \
  --net host \
  --runtime nvidia \
  --gpus all \
  -e ACCEPT_EULA=Y \
  -e DISPLAY=$DISPLAY \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v isaac-cache-kit-232:/isaac-sim/kit/cache \
  -v isaac-cache-ov-232:/root/.cache/ov \
  -v <your_isaaclab_repo_path>:/workspace/isaaclab \
  --name isaac-lab-232 \
  isaac-lab-teleop:2.3.2
```

**命名卷**：`isaac-cache-kit-232` → `/isaac-sim/kit/cache`；`isaac-cache-ov-232` → `/root/.cache/ov`，持久化缓存。

**DISPLAY / GUI**：依赖 **2.2** 的 `xhost`。任务二 **`handtracking` + `--enable_cameras`** 勿再加 `--headless`（易出现 SyntheticData `LdrColorSD` 等报错及空图像张量）。`OMNI_KIT_ALLOW_ROOT=1` 供 root 跑 Kit。

**方式 A：`ln -sfn` 建立 `_isaac_sim`（每个新容器执行一次）**

```bash
docker exec -it isaac-lab-232 /bin/bash
cd /workspace/isaaclab
ln -sfn /isaac-sim _isaac_sim
test -f _isaac_sim/VERSION && head -n1 _isaac_sim/VERSION
```

能打印版本号即正常。方式 B 通常跳过本段。

等待 CloudXR 就绪：

```
The NVIDIA(TM) CloudXR(TM) Runtime service has started.
```

**方式 B：`docker cp` 示例（任务二常见补丁文件）**  
容器须已运行，在**宿主机**执行：

```bash
ISAAC_HOST="<your_isaaclab_repo_path>"
REL="source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/galbot/stack_rmp_rel_env_cfg.py"
docker cp "${ISAAC_HOST}/${REL}" "isaac-lab-232:/workspace/isaaclab/${REL}"
```

`yujie-dev` 上其它改动对**相同相对路径**重复 `docker cp`。仅改 `.py` 时重跑 `record_demos` 即可，不必重建镜像。

---

## 3. 配置服务端防火墙

Pico 4 Ultra 使用 HTTPS/WSS 模式连接，需要开放以下端口：

```bash
# CloudXR Runtime 信令端口 (WebRTC)
sudo ufw allow 49100/tcp

# CloudXR Runtime 媒体流端口范围 (UDP)
sudo ufw allow 47998:48012/udp

# WSS 代理端口 (HTTPS 模式)
sudo ufw allow 48322/tcp
```

端口说明：

| 端口 | 协议 | 用途 |
|------|------|------|
| 49100 | TCP | CloudXR WebRTC 信令 |
| 47998-48012 | UDP | CloudXR 媒体流传输 |
| 48322 | TCP | WSS 代理 (SSL 加密信令) |

---

## 4. 启动 WSS 代理

按照步骤 1.3 构建好 HAProxy 镜像后，启动 WSS 代理容器：

```bash
docker run -d --name wss-proxy \
  --network host \
  -e BACKEND_HOST=localhost \
  -e BACKEND_PORT=49100 \
  -e PROXY_PORT=48322 \
  websocket-ssl-proxy
```

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `BACKEND_HOST` | localhost | CloudXR Runtime 地址 |
| `BACKEND_PORT` | 49100 | CloudXR WebRTC 信令端口 |
| `PROXY_PORT` | 48322 | WSS 对外暴露端口 |

---

## 5. 启动 HTTPS Dev Server

在服务端另开一个终端，启动 cloudxr-js 的 HTTPS dev server：

```bash
cd <cloudxr-js-early-access_6.0.2-beta-pid 解压目录>
cd isaac
npm run dev-server:https
```

此时 Pico 4 Ultra 可以通过浏览器访问 `https://<服务器IP>:<端口>` 连接到 CloudXR。

---

## 6. 进入容器并按任务运行遥操作

另开一个终端，exec 进入容器：

```bash
docker exec -it isaac-lab-232 /bin/bash
```

**前提**：已按 **2.2** 开放 X11；已按 **2.3** 启动容器（含 `DISPLAY`）。若使用 **2.3 方式 A（mount）**，须完成 **`ln -sfn` → `_isaac_sim`**；若使用 **方式 B**，须已 **`docker cp`** 打入 **2.3** 所需补丁。GUI 与 headless 见 **2.3**。

**Headless 与任务对应关系**：

- **任务一（GR1 PickPlace）**：`record_demos` 示例带 `--headless`（该任务默认不录相机观测时可常用）。若出现与 **2.3** 类似的 XR /渲染报错，可去掉 `--headless` 改走 GUI。
- **任务二（Galbot Visuomotor）**：需 **`--enable_cameras`** 写入图像，**不要**加 `--headless`（见 **2.3**）。
- **任务三（Franka IK Abs + motion controllers）**：与任务一类似走 **`--headless`** 即可；无需 `--enable_pinocchio`（微分 IK）。

以下命令均在容器内 `/workspace/isaaclab` 执行（`./isaaclab.sh`）。

### 任务一：`Isaac-PickPlace-GR1T2-Abs-v0`

#### 仅遥操（不录制）

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --teleop_device handtracking \
  --enable_pinocchio \
  --info
```

#### 录制 HDF5

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --teleop_device handtracking \
  --enable_pinocchio \
  --info \
  --headless \
  --dataset_file ./datasets/pickplace_gr1t2_handtracking.hdf5 \
  --num_demos 0 \
  --num_success_steps 10
```

### 任务二：`Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0`（录制含相机）

运行前须已让容器内代码为 [yujie-dev](https://github.com/billamiable/IsaacLab/tree/yujie-dev)：见 **2.3** — **方式 A** 为 mount + `_isaac_sim`；**方式 B** 为无 mount + **`docker cp`**。

Galbot RmpFlow 相对模式需设置 `USE_RELATIVE_MODE`；显式指定 GPU 与渲染质量，并开启相机写入观测。

```bash
USE_RELATIVE_MODE=true ./isaaclab.sh -p scripts/tools/record_demos.py \
  --device cuda:0 \
  --task Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0 \
  --teleop_device handtracking \
  --enable_cameras \
  --info \
  --rendering_mode quality \
  --dataset_file ./datasets/galbot_left_visuomotor_handtracking.hdf5 \
  --num_demos 0 \
  --num_success_steps 10
```

**任务二为非 headless，会弹出 Isaac Sim 主窗口。** 使用 `handtracking` 前，需在界面里手动进入 XR（与下图一致）：

1. 在窗口**底部面板**选中 **「AR」** 标签页。  
2. 确认 **Selected Output Plugin** 为 **OpenXR**（**OpenXR Runtime** 一般为系统运行时）。  
3. 点击 **「Start AR」**（带头显图标的大按钮），启动后再用 Pico 浏览器连接 CloudXR 进行手部追踪。

<img src="./isaac-lab-xr-open.png" alt="Isaac Sim AR 面板：Start AR（OpenXR）" width="920" />

| 参数 | 说明 |
|------|------|
| `--dataset_file` | 录制 HDF5 在容器内的保存路径（导出见第 7 节） |
| `--num_demos 0` | 不限制 demo 条数，手动结束 |
| `--num_success_steps 10` | 连续10 步满足成功条件后判定当前 demo 成功 |
| `--enable_cameras` | **任务二**需要，用于 policy 观测中的 RGB（及录制进 HDF5） |

通过 Pico 4 Ultra 连接 CloudXR 后即可手部追踪遥操作；**原始录制**落在容器内 `./datasets/*.hdf5`。需要 MP4 时可在容器内再跑 **9.2** 的脚本，输出到例如 `./videos_for_cosmos/`。

### 任务三：`Isaac-Stack-Cube-Franka-IK-Abs-v0`（右手柄 + 扳机夹爪）

[yujie-dev](https://github.com/billamiable/IsaacLab/tree/yujie-dev) 在 **`stack_ik_abs_env_cfg`** 中增加了与 Isaac Lab 3 / IsaacTeleop **语义对齐** 的 **`motion_controllers`** 设备：`OpenXRDeviceCfg` + 右手控制器绝对位姿（Se3Abs）+ **扳机优先、手部 pinch 兜底** 的夹爪逻辑（无需 Pink / `--enable_pinocchio`）。同一任务仍保留 **`handtracking`** 设备键。

| 项目 | 说明 |
|------|------|
| 手柄 | **右手**（`CONTROLLER_RIGHT`）驱动末端绝对位姿 |
| 夹爪 | **右手扳机（trigger）**：按下超过默认阈值（0.5）闭合；未超过张开；无手柄数据时退化为右手 pinch |
| 叠加方块顺序 | 底层蓝 `cube_1`、中层红 `cube_2`、顶层绿 `cube_3`（成功判定按实体名，非颜色分类算法） |

#### 录制 HDF5（motion controllers）

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
  --teleop_device motion_controllers \
  --info \
  --headless \
  --dataset_file ./datasets/franka_stack_cube_ik_abs_motion_controllers.hdf5 \
  --num_demos 0 \
  --num_success_steps 10
```

---

## 7. 导出录制数据

录制完成后，在**宿主机**上进入希望存放数据的目录，将 HDF5（以及若已生成的视频目录）从容器拷出：

```bash
docker cp isaac-lab-232:/workspace/isaaclab/datasets/ ./datasets/
docker cp isaac-lab-232:/workspace/isaaclab/videos_for_cosmos/ ./videos_for_cosmos/
```

第二行仅在容器内已执行过 **9.2**、且目录已存在时需要；若尚未生成 `videos_for_cosmos`，可跳过该行（执行会报错属正常）。

**说明**：`record_demos` 一般会创建 `./datasets/`；若首次报错无目录，可在容器内先执行 `mkdir -p /workspace/isaaclab/datasets`。

---

## 8. 完整操作流程总结

```
服务端终端 1: 启动 Isaac Lab 容器 (CloudXR 自动启动)
     │
服务端终端 2: 启动 WSS 代理 (docker run，镜像名与 **4** 中一致，如 websocket-ssl-proxy)
     │
服务端终端 3: 启动 cloudxr-js HTTPS Dev Server (npm run dev-server:https)
     │
服务端终端 4: docker exec 进入容器，按任务运行 teleop_se3_agent / record_demos.py
     │
Pico 4 Ultra: 浏览器访问 https://<服务器IP> → 连接 CloudXR → 手部追踪遥操作
     │
可选:        容器内对 HDF5 运行 hdf5_to_mp4.py → 得到按相机拆分的 MP4
     │
录制完成后:   docker cp 导出 datasets/（及 videos_for_cosmos/）
```

---

## 9. 录制数据格式与目录结构

### 9.1 总览：HDF5 + 可选视频

| 类型 | 来源 | 典型位置（容器内） | 说明 |
|------|------|-------------------|------|
| **HDF5** | `record_demos.py` | `./datasets/*.hdf5` | 主数据：轨迹、状态、观测；任务二含 RGB 帧；任务三与任务一类似以状态/向量观测为主（默认无 Visuomotor 相机键） |
| **MP4** | `scripts/tools/hdf5_to_mp4.py` | `./videos_for_cosmos/`（目录可自定） | 从 HDF5 的 `obs/<相机键>` 导出，每条 demo、每个相机一个文件 |

HDF5 内 demo 路径均为 `data/demo_0`、`data/demo_1`、…。

### 9.2 从 HDF5 导出 MP4（任务二多机位示例）

脚本路径：`scripts/tools/hdf5_to_mp4.py`。**任务二**相机键名为 `ego_cam`、`left_wrist_cam`、`right_wrist_cam`，需用 `--input_keys` 指定（默认的 `table_cam` / `wrist_cam` 不适用）。参数名为 **`--framerate`**（不是 `--fps`）。

容器内示例（在 `/workspace/isaaclab`）：

```bash
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
  --input_file ./datasets/galbot_left_visuomotor_handtracking.hdf5 \
  --output_dir ./videos_for_cosmos \
  --input_keys ego_cam left_wrist_cam right_wrist_cam \
  --framerate 30
```

输出文件命名：`demo_<序号>_<相机键>.mp4`（例如 `demo_0_ego_cam.mp4`）。可选 `--video_height` / `--video_width`（默认会放大到 704×1280；若需保持 256×256 可显式指定）。

**任务一**若 HDF5 中观测键与 Franka/GR1 示例一致，可使用脚本默认的 `--input_keys`，或按实际 `obs/` 下的数据集名称自行传入。

### 9.3 HDF5：`data/demo_N/` 结构（任务一 GR1 PickPlace 示例）

任务一输出例如 `pickplace_gr1t2_handtracking.hdf5`。每个 `demo_N` 大致如下（具体键名以实际文件为准）：

```
data/
├── env_args          (环境配置: env_name, dt, decimation, render_interval, num_envs)
├── total             (总帧数)
└── demo_N/
    ├── num_samples   (当前 demo 帧数)
    ├── success       (是否成功)
    ├── actions                          (N, 36)   原始动作
    ├── processed_actions                (N, 36)   处理后的动作
    ├── obs/
    │   ├── actions                      (N, 36)   观测中的动作
    │   ├── hand_joint_state             (N, 22)   手部关节状态
    │   ├── head_joint_state             (N, 3)    头部关节状态
    │   ├── left_eef_pos                 (N, 3)    左末端执行器位置
    │   ├── left_eef_quat                (N, 4)    左末端执行器四元数
    │   ├── right_eef_pos                (N, 3)    右末端执行器位置
    │   ├── right_eef_quat               (N, 4)    右末端执行器四元数
    │   ├── robot_joint_pos              (N, 54)   机器人全部关节位置
    │   ├── robot_links_state            (N, 55, 13) 55个link状态
    │   ├── robot_root_pos               (N, 3)    机器人根节点位置
    │   ├── robot_root_rot               (N, 4)    机器人根节点旋转
    │   ├── object                       (N, 13)   物体完整状态
    │   ├── object_pos                   (N, 3)    物体位置
    │   └── object_rot                   (N, 4)    物体旋转
    ├── initial_state/
    │   ├── articulation/robot/
    │   │   ├── joint_position           (1, 54)
    │   │   ├── joint_velocity           (1, 54)
    │   │   ├── root_pose                (1, 7)
    │   │   └── root_velocity            (1, 6)
    │   └── rigid_object/object/
    │       ├── root_pose                (1, 7)
    │       └── root_velocity            (1, 6)
    └── states/                          (与 initial_state 结构相同，记录全部 N 帧)
```

### 9.4 HDF5：`data/demo_N/` 结构（任务二 Galbot Visuomotor 示例）

任务二输出例如 `galbot_left_visuomotor_handtracking.hdf5`。在 **yujie-dev** policy 含相机时，`obs/` 中除向量观测外，还有 RGB（`uint8`，形状 `(N, 256, 256, 3)`）：

```
data/demo_N/
├── actions, processed_actions
├── obs/
│   ├── ego_cam, left_wrist_cam, right_wrist_cam   (N, 256, 256, 3)  uint8  ← 与 MP4 导出对应
│   ├── joint_pos, joint_vel, eef_pos, eef_quat, gripper_pos, object, ...
│   └── ...（cube 位置/姿态等，以实际录制为准）
├── initial_state/ (articulation/robot, rigid_object/cube_*)
└── states/          (同上结构，逐帧)
```

图像已存在于 HDF5 时，**MP4 为可选衍生格式**，便于目视检查或接入只接受视频的下游（如 Cosmos 增广流水线）。

### 关键维度说明（任务一 GR1）

| 维度 | 含义 |
|------|------|
| 36 | 动作空间 (左臂 + 右臂 + 双手) |
| 54 | GR1T2 全部关节数 |
| 22 | 双手关节自由度 |
| 3 | 头部关节自由度 (yaw/pitch/roll) |
| 55 | 机器人 link 数量 |
| 13 | 单个刚体完整状态 (pos3 + quat4 + lin_vel3 + ang_vel3) |
| 7 | 位姿 (pos3 + quat4) |
| 6 | 速度 (lin_vel3 + ang_vel3) |
