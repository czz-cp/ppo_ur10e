# UR10e PPO 多目标最优轨迹规划

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》的完整实现，使用PPO算法进行UR10e机械臂的多目标轨迹规划。

## 📋 项目概述

本项目实现了一个完整的深度强化学习框架，专门用于UR10e机械臂的多目标最优轨迹规划。核心特性包括：

### 🎯 论文核心实现

1. **25维状态空间设计**：
   - `θ_start`: 起始关节角（6维）
   - `p_c`: 末端当前位置（3维）
   - `θ_end`: 目标关节角（6维）
   - `p_d`: 末端目标位置（3维）
   - `d_e`: 当前位置与目标位置的欧氏距离（1维）
   - 当前关节速度（6维）

2. **6维动作空间**：
   - 关节角速度 `\dot{θ}`（6维）

3. **多目标奖励函数**：
   ```
   r(s_t, a_t) = r_a^t + r_s^t + r_e^t + r_ex^t
   ```
   - **精度奖励** `r_a^t`: 使用指数函数放大误差
   - **平滑性奖励** `r_s^t`: 惩罚速度和加速度
   - **能量消耗奖励** `r_e^t`: 惩罚功率消耗
   - **额外奖励** `r_ex^t`: 稀疏成功奖励

4. **衰减回合机制**（论文创新点）：
   - 自适应课程学习
   - 根据成功率动态调整最大episode长度
   - 提高训练效率

## 🚀 快速开始

### 环境要求

```bash
# Python依赖
pip install torch numpy matplotlib pandas pyyaml scipy

# MuJoCo依赖（如果使用MuJoCo仿真）
pip install mujoco
```

### 使用方法

1. **启动训练**：
```bash
cd ppo_ur10e
python run.py
```

2. **或者直接训练**：
```bash
python train.py
```

3. **配置文件**：
编辑 `config.yaml` 文件来自定义训练参数。

## 📁 文件结构

```
ppo_ur10e/
├── README.md                 # 项目说明
├── run.py                   # 启动脚本
├── config.yaml             # 配置文件
├── utils.py                 # 工具函数
├── ur10e_env.py            # 环境实现（25维状态空间）
├── ppo.py                  # PPO算法实现
├── train.py                # 主训练程序
├── checkpoints/            # 模型检查点
├── csv_output/            # 训练数据
├── plots/                # 训练曲线
├── trajectories/          # 轨迹可视化
└── logs/                 # 日志文件
```

## ⚙️ 配置参数

### 主要配置项

```yaml
# 环境配置
env:
  max_steps: 1000              # 每个episode最大步数
  enable_rendering: false      # 训练时关闭渲染
  action_bound: 0.0189         # 动作限制（弧度/步）

# PPO配置
ppo:
  lr_actor: 3e-4               # Actor学习率
  lr_critic: 1e-3              # Critic学习率
  clip_eps: 0.2                # PPO裁剪参数
  gamma: 0.99                  # 折扣因子
  epochs: 10                   # 训练轮数

# 多目标奖励函数
reward:
  accuracy:
    weight: 10.0               # 精度权重
    sigma: 2.0                 # 精度锐度参数
    threshold: 0.005           # 成功阈值
  smoothness:
    weight: 0.1                # 平滑性权重
    lambda_vel: 1.0           # 速度惩罚权重
    lambda_acc: 0.5           # 加速度惩罚权重
  energy:
    weight: 0.01              # 能量权重

# 衰减回合机制
decay_episode:
  enabled: true                # 启用衰减回合
  success_threshold: 0.7       # 成功率阈值
  decay_window: 200            # 计算窗口
```

## 📊 训练输出

### 训练过程中会生成以下文件：

1. **模型检查点** (`checkpoints/`):
   - `ppo_model_episode_500.pt`
   - `ppo_model_episode_1000.pt`
   - `ppo_model_final.pt`

2. **训练数据** (`csv_output/`):
   - `training_data.csv`: 完整训练记录
   - 包含奖励、长度、误差等信息

3. **训练曲线** (`plots/`):
   - Episode奖励曲线
   - 成功率变化
   - 位置误差变化
   - 衰减回合统计

4. **轨迹可视化** (`trajectories/`):
   - 3D轨迹图
   - 关节角度变化

## 🎯 核心创新点

### 1. 多目标奖励函数设计

```python
def _compute_multi_objective_reward(self, action, joint_angles, joint_velocities, end_pos, target_pos):
    # 精度奖励（指数函数放大误差）
    pos_error = np.linalg.norm(target_pos - end_pos)
    accuracy_reward = -weight * np.exp(sigma * pos_error)

    # 平滑性奖励（惩罚速度和加速度）
    velocity_penalty = np.sum(joint_velocities ** 2) * lambda_vel
    smoothness_reward = -weight * (velocity_penalty + acceleration_penalty)

    # 能量消耗奖励
    power_approx = action * np.abs(joint_angles)
    energy_reward = -weight * np.sum(power_approx ** 2)

    # 稀疏成功奖励
    if pos_error < threshold:
        extra_reward = 10.0
    else:
        extra_reward = 10.0 / (1.0 + 10.0 * pos_error)

    return accuracy_reward + smoothness_reward + energy_reward + extra_reward
```

### 2. 衰减回合机制

```python
def update_decay_episode_mechanism(self, episode_reward, final_pos_error):
    # 判断是否成功
    is_success = final_pos_error < threshold
    self.success_history.append(is_success)

    # 计算成功率
    recent_success_rate = sum(self.success_history[-window:]) / window

    # 如果成功率高，减少最大步数
    if recent_success_rate >= self.success_threshold:
        self.current_max_steps = max(self.min_max_steps,
                                     self.current_max_steps - decay_step)
```

### 3. 25维状态空间实现

```python
def _get_state(self) -> np.ndarray:
    """
    25维状态向量：
    [θ_start(6), p_c(3), θ_end(6), p_d(3), d_e(1), current_joint_velocities(6)]
    """
    state = np.concatenate([
        self.start_joint_angles,        # 起始关节角（6维）
        current_end_pos,               # 末端当前位置（3维）
        self.target_joint_angles,      # 目标关节角（6维）
        self.target_pos,               # 目标位置（3维）
        [distance_to_target],          # 位置距离（1维）
        current_joint_velocities       # 当前关节速度（6维）
    ])
    return state
```

## 🔧 自定义扩展

### 添加新的奖励分量

在 `ur10e_env.py` 的 `_compute_multi_objective_reward` 方法中添加：

```python
# 添加新的奖励分量
new_reward = -weight * np.sum(action ** 2)  # 示例：惩罚动作幅度
total_reward = accuracy_reward + smoothness_reward + energy_reward + extra_reward + new_reward
```

### 修改网络架构

在 `ppo.py` 中修改 `ActorNetwork` 和 `CriticNetwork`：

```python
class ActorNetwork(nn.Module):
    def __init__(self, state_dim=25, action_dim=6, hidden_dim=512):  # 增加隐藏层维度
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # 添加更多层
            nn.ReLU(),
            # ... 更多层
        )
```

## 📈 性能指标

训练完成后，系统会输出以下性能指标：

1. **成功率**: 达到目标阈值（<5mm）的比例
2. **平均轨迹长度**: 完成任务的平均步数
3. **轨迹平滑度**: 基于加速度的平滑度指标
4. **位置误差**: 最终位置的定位精度
5. **能量效率**: 轨迹的能量消耗

## 🔍 调试和可视化

### 启用渲染模式

在 `config.yaml` 中设置：
```yaml
env:
  enable_rendering: true
```

### 查看详细日志

训练日志会保存在 `logs/` 目录下，包含：
- 训练进度
- 奖励分量分析
- 衰减回合统计
- 错误信息

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于MIT许可证开源。

## 📚 参考文献

论文：《基于深度强化学习的机械臂多目标最优轨迹规划》

---

**注意**：本项目需要MuJoCo或类似物理引擎支持。确保已正确安装相关依赖。