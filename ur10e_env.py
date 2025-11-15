"""
UR10e PPO 多目标最优轨迹规划环���

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》
实现了论文中设计的25维状态空间和多目标奖励函数

状态空间设计（16维 - RL-PID混合控制）：
- θ_start: 起始关节角（6维）
- p_c: 末端当前位置（3维）
- θ_end: 目标关节角（6维）
- p_d: 末端目标位置（3维）
- d_e: 当前位置与目标���置的欧氏距离（1维）

动作空间设计（6维）：
- 关节角速度 \dot{θ}（6维）

奖励函数设计：
r(s_t, a_t) = r_a^t + r_s^t + r_e^t + r_ex^t
- r_a^t: 精度奖励（指数函数）
- r_s^t: 平滑性奖励（速度和加速度惩罚）
- r_e^t: 能量消耗奖励
- r_ex^t: 额外奖励（稀疏奖励）

创新点：
- 衰减回合机制（自适应课程学习）
"""

import numpy as np
import torch
import mujoco as mj
from mujoco.glfw import glfw
from scipy.spatial.transform import Rotation as R
import math
from typing import Tuple, Dict, Any, Optional
from ur10e_kinematics_fixed import UR10eKinematicsFixed


class UR10ePPOEnv:
    """
    UR10e PPO轨迹规划环境

    实现论文中的25维状态空间设计和多目标奖励函数
    """

    def __init__(self, xml_path: str = './universal_robots_ur10e/ur10e_mujoco/scene.xml',
                 max_steps: int = 1000, enable_rendering: bool = False, config: Dict = None):
        # 基础参数
        self.xml_path = xml_path
        self.max_steps = max_steps
        self.enable_rendering = enable_rendering

        # 加载配置
        self.config = config or {}

        # UR10e关节限制（弧度）- 从配置加载
        if 'joint_limits' in self.config.get('env', {}):
            self.joint_limits = np.array(self.config['env']['joint_limits'])
        else:
            self.joint_limits = np.array([
                [-3.14159, 3.14159],  # shoulder_pan_joint
                [-3.14159, 3.14159],  # shoulder_lift_joint
                [-3.14159, 3.14159],  # elbow_joint
                [-3.14159, 3.14159],  # wrist_1_joint
                [-3.14159, 3.14159],  # wrist_2_joint
                [-3.14159, 3.14159]   # wrist_3_joint
            ])

        # 动作空间限制（关节角速度，弧度/步）- 从配置加载
        self.action_bound = self.config.get('env', {}).get('action_bound', 0.03)

        # 动作执行时间步长（秒）- 将角速度转换为角度增量
        self.dt = self.config.get('env', {}).get('dt', 0.01)

        # 目标位置范围（米）- 从配置加载
        if 'target_pos_range' in self.config.get('env', {}):
            self.target_pos_range = self.config['env']['target_pos_range']
        else:
            self.target_pos_range = {
                'x': [0.3, 1.2],
                'y': [-0.6, 0.6],
                'z': [0.1, 0.8]
            }

        # 奖励函数参数（基于论文设计）- 从配置加载
        if 'reward' in self.config:
            self.reward_config = self.config['reward']
        else:
            self.reward_config = {
                'accuracy': {
                    'weight': 1.5,      # w_a
                    'sigma': 1.0,       # σ_a
                    'threshold': 0.005   # 成功阈值
                },
                'smoothness': {
                    'weight': 0.1,      # w_s
                    'lambda_vel': 1.0,  # λ_v
                    'lambda_acc': 0.5   # λ_acc
                },
                'energy': {
                    'weight': 0.01      # w_e
                },
                'extra': {
                    'success_reward': 20.0
                }
            }

        # 衰减回合机制参数
        self.decaying_episode_enabled = True
        self.current_max_steps = max_steps
        self.min_max_steps = 200
        self.success_threshold = 0.7
        self.success_history = []
        self.decay_window = 200

        # 内部状态
        self.model = None
        self.data = None
        self.viewer = None
        self.cam = None
        self.opt = None
        self.scene = None
        self.context = None

        # 状态变量
        self.current_step = 0
        self.start_joint_angles = None
        # self.target_joint_angles = None  # RL-PID混合控制不需要
        self.target_pos = None
        self.prev_joint_angles = None
        self.prev_joint_velocities = None

        # 历史数据（用于计算加速度）
        self.joint_history = []

        # 初始化运动学解算器
        # 初始化运动学 - 使用官方UR代码实现
        self.kinematics = UR10eKinematicsFixed()

        # 初始���环境
        self._init_environment()

    def _init_environment(self):
        """初始化MuJoCo环境"""
        # 加载模型
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)

        # 初始化渲染
        if self.enable_rendering:
            self._init_rendering()

        # 设置PD控制器
        self._setup_pd_controller()

        print(f"UR10e PPO环境初始化完成")
        print(f"状态空间: 16维 (RL-PID混合控制状态)")
        print(f"动作空间: 3维 (PID参数调度)")
        print(f"最大步数: {self.max_steps}")
        print(f"奖励函数: 多目标 (精度 + 平滑性 + 能耗)")

    def _init_rendering(self):
        """初始化渲染环境"""
        glfw.init()
        self.viewer = glfw.create_window(1200, 900, "UR10e PPO Trajectory Planning", None, None)
        glfw.make_context_current(self.viewer)
        glfw.swap_interval(1)

        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # 设置相机
        self.cam.azimuth = 118.0
        self.cam.elevation = -52.8
        self.cam.distance = 2.84
        self.cam.lookat = np.array([-0.024, 0.011, 0.241])

        # 设置回调
        glfw.set_key_callback(self.viewer, self._keyboard_callback)
        glfw.set_cursor_pos_callback(self.viewer, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.viewer, self._mouse_button_callback)
        glfw.set_scroll_callback(self.viewer, self._scroll_callback)

    def _setup_pd_controller(self):
        """设置PD控制器（基于原DDPG实现优化）"""
        # 启用所有关节的力矩伺服器
        for i in range(6):
            self.model.actuator_gainprm[i, 0] = 1.0  # 启用力矩控制

        # PD控制参数（基于原DDPG的分段设计）
        # 关节1-3（大关节）：P=6500, D=200
        # 关节4-6（腕部关节）：P=5000, D=200
        pd_params = [
            (6500.0, 200.0),  # shoulder_pan_joint
            (6500.0, 200.0),  # shoulder_lift_joint
            (6500.0, 200.0),  # elbow_joint
            (5000.0, 200.0),  # wrist_1_joint
            (5000.0, 200.0),  # wrist_2_joint
            (5000.0, 200.0),  # wrist_3_joint
        ]

        for i, (p_gain, d_gain) in enumerate(pd_params):
            self.model.actuator_biasprm[i, 1] = p_gain  # P增益
            self.model.actuator_biasprm[i, 2] = d_gain  # D增益

        print("🔧 PD控制器参数设置完成:")
        print("  关节1-3: P=6500, D=200 (大关节)")
        print("  关节4-6: P=5000, D=200 (腕部关节)")

    def _pd_control(self, target_angles: np.ndarray) -> np.ndarray:
        """
        PD控制器实现（基于原DDPG的控制逻辑）

        Args:
            target_angles: [6] 目标关节角度

        Returns:
            control_torques: [6] 控制力矩
        """
        control_torques = np.zeros(6)

        # 获取当前状态
        current_angles = self.data.qpos[:6].copy()
        current_velocities = self.data.qvel[:6].copy()

        # PD控制（基于原DDPG的分段参数）
        for i in range(6):
            if i < 3:  # 关节1-3
                p_gain = 6500.0
            else:       # 关节4-6
                p_gain = 5000.0

            d_gain = 200.0  # 所有关节使用相同的D增益

            # PD控制公式：τ = -K_p * e - K_d * ẋ
            error = target_angles[i] - current_angles[i]
            error_vel = -current_velocities[i]  # 期望速度为0

            control_torques[i] = -p_gain * error - d_gain * error_vel

        return control_torques

    def _rl_pid_control(self, current_angles: np.ndarray, current_velocities: np.ndarray,
                       kp_scale: float, kd_scale: float, ki_enable: float) -> np.ndarray:
        """
        RL调度的PID控制器 - 重新设计

        Args:
            current_angles: [6] 当前关节角度
            current_velocities: [6] 当前关节速度
            kp_scale: P增益缩放因子 [-0.5, 0.5]
            kd_scale: D增益缩放因子 [-0.5, 0.5]
            ki_enable: 积分控制启用标志 [0, 1]

        Returns:
            control_torques: [6] 控制力矩
        """
        control_torques = np.zeros(6)

        # 初始化积分误差累积（如果需要）
        if not hasattr(self, 'error_integral'):
            self.error_integral = np.zeros(6)

        # 基于当前位置到目标位置的启发式控制
        # 使用雅可比的近似版本：将末端位置误差映射到关节误差
        if hasattr(self, 'target_pos') and self.target_pos is not None:
            current_end_pos = self.data.site_xpos[0].copy()
            end_effector_error = self.target_pos - current_end_pos

            # 简化的雅可比映射：将末端误差分配到各关节
            # 这是一个启发式方法，实际应用中可以使用更精确的雅可比计算
            position_error = np.zeros(6)

            # 将XYZ误差映射到关节空间（简化的启发式映射）
            position_error[0] = end_effector_error[0] * 0.5  # shoulder_pan
            position_error[1] = end_effector_error[1] * 0.5  # shoulder_lift
            position_error[2] = end_effector_error[2] * 0.5  # elbow
            position_error[3] = (end_effector_error[0] + end_effector_error[1]) * 0.3  # wrist_1
            position_error[4] = (end_effector_error[1] + end_effector_error[2]) * 0.3  # wrist_2
            position_error[5] = np.linalg.norm(end_effector_error) * 0.2  # wrist_3
        else:
            # 如果没有目标，只做阻尼控制
            position_error = np.zeros(6)

        for i in range(6):
            # 基础PID参数
            if i < 3:  # 关节1-3
                base_p_gain = 6500.0
            else:       # 关节4-6
                base_p_gain = 5000.0
            base_d_gain = 200.0
            base_i_gain = 10.0  # 小的积分增益

            # RL调度参数调整
            p_gain = base_p_gain * (1.0 + kp_scale)  # ±50%调整范围
            d_gain = base_d_gain * (1.0 + kd_scale)  # ±50%调整范围
            i_gain = base_i_gain * ki_enable        # 0或启用

            # 速度误差（期望速度为0）
            velocity_error = -current_velocities[i]

            # 更新积分误差（抗饱和）
            if ki_enable > 0.5:  # 启用积分控制
                self.error_integral[i] += position_error[i] * self.dt
                # 积分抗饱和
                self.error_integral[i] = np.clip(self.error_integral[i], -0.5, 0.5)
            else:
                self.error_integral[i] *= 0.95  # 缓慢衰减积分项

            # PID控制公式
            control_torques[i] = (
                p_gain * position_error[i] +          # 比例项
                d_gain * velocity_error +              # 微分项
                i_gain * self.error_integral[i]        # 积分项
            )

        return control_torques

    def _compute_hybrid_control_reward(self, action: np.ndarray, current_angles: np.ndarray,
                                     current_velocities: np.ndarray, current_end_pos: np.ndarray,
                                     target_pos: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        计算混合控制奖励函数

        Args:
            action: 3维PID调度动作
            current_angles: 当前关节角度
            current_velocities: 当前关节速度
            current_end_pos: 当前末端位置
            target_pos: 目标位置

        Returns:
            reward: 总奖励值
            reward_components: 各项奖励组件
        """
        # 1. 控制精度奖励 (最重要)
        position_error = np.linalg.norm(target_pos - current_end_pos)
        accuracy_reward = -self.reward_config['accuracy']['weight'] * position_error

        # 2. 控制稳定性奖励 (参数变化不能太剧烈)
        kp_scale, kd_scale, ki_enable = action
        stability_reward = -self.reward_config.get('stability', {}).get('weight', 0.1) * (
            abs(kp_scale) + abs(kd_scale)
        )

        # 3. 响应速度奖励 (鼓励快速收敛)
        if hasattr(self, 'prev_position_error'):
            error_change = self.prev_position_error - position_error
            speed_reward = self.reward_config.get('speed', {}).get('weight', 0.05) * max(0, error_change)
        else:
            speed_reward = 0.0
        self.prev_position_error = position_error

        # 4. 能耗效率奖励
        energy_cost = np.sum(current_velocities**2)
        energy_reward = -self.reward_config.get('energy', {}).get('weight', 0.001) * energy_cost

        # 总奖励
        total_reward = accuracy_reward + stability_reward + speed_reward + energy_reward

        # 成功奖励
        if position_error < self.reward_config['accuracy']['threshold']:
            total_reward += self.reward_config['extra']['success_reward']

        reward_components = {
            'accuracy': accuracy_reward,
            'stability': stability_reward,
            'speed': speed_reward,
            'energy': energy_reward,
            'total': total_reward
        }

        return total_reward, reward_components

    def _normalize_joint_positions(self, positions):
        """关节位置归一化到[-1, 1]（基于UR10e实际限制）

        Args:
            positions: 关节角度 [6] in rad/s

        Returns:
            normalized: 归一化后的关节角度 [6] in [-1, 1]
        """
        # UR10e实际关节位置限制 (rad)
        max_positions = np.array([
            6.283,  # shoulder_pan: ±360° = ±2π rad
            6.283,  # shoulder_lift: ±360° = ±2π rad
            3.142,  # elbow_joint: ±180° = ±π rad (人工限制)
            6.283,  # wrist_1: ±360° = ±2π rad
            6.283,  # wrist_2: ±360° = ±2π rad
            6.283   # wrist_3: ±360° = ±2π rad
        ])

        # 避免除零
        normalized = np.divide(positions, max_positions,
                             out=np.zeros_like(positions),
                             where=max_positions!=0)

        return np.clip(normalized, -1.0, 1.0)

    def _normalize_joint_velocities(self, velocities):
        """关节速度归一化到[-1, 1]（基于UR10e实际限制）

        Args:
            velocities: 关节速度 [6] in rad/s

        Returns:
            normalized: 归一化后的关节速度 [6] in [-1, 1]
        """
        # UR10e实际关节速度限制 (rad/s)
        max_velocities = np.array([
            2.094,  # shoulder_pan_joint: 120°/s = 2.094 rad/s
            2.094,  # shoulder_lift_joint: 120°/s = 2.094 rad/s
            3.142,  # elbow_joint: 180°/s = π rad/s
            3.142,  # wrist_1_joint: 180°/s = π rad/s
            3.142,  # wrist_2_joint: 180°/s = π rad/s
            3.142   # wrist_3_joint: 180°/s = π rad/s
        ])

        # 避免除零
        normalized = np.divide(velocities, max_velocities,
                             out=np.zeros_like(velocities),
                             where=max_velocities!=0)

        return np.clip(normalized, -1.0, 1.0)

    def _normalize_position_error(self, position_error):
        """位置误差归一化到[-1, 1]

        Args:
            position_error: 位置误差 [3] in [-1, 1] meters

        Returns:
            normalized: 归一化后的位置误差 [3] in [-1, 1]
        """
        return np.clip(position_error / 1.0, -1.0, 1.0)  # 假设最大误差1米

    def _normalize_distance(self, distance):
        """距离归一化到[0, 1]

        Args:
            distance: 欧式距离 in [0, 2] meters

        Returns:
            normalized: 归一化后的距离 in [0, 1]
        """
        return np.clip(distance / 2.0, 0.0, 1.0)  # 假设最大距离2米

    def _keyboard_callback(self, window, key, scancode, act, mods):
        """键盘回调"""
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def _mouse_button_callback(self, window, button, act, mods):
        """鼠标按钮回调"""
        # 简化实现
        pass

    def _mouse_move_callback(self, window, xpos, ypos):
        """鼠标移动回调"""
        # 简化实现
        pass

    def _scroll_callback(self, window, xoffset, yoffset):
        """鼠标滚轮回调"""
        # 简化实现
        pass

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns:
            state: 16维状态向量 (RL-PID混合控制)
        """
        # 重置MuJoCo状态
        mj.mj_resetData(self.model, self.data)

        # 随机生成起始关节角度
        self.start_joint_angles = self._sample_random_joint_angles()

        # 随机生成目标位置
        self.target_pos = self._sample_random_target_pos()

        # RL-PID混合控制：不需要目标关节角度，RL会动态调整PID参数
        # self.target_joint_angles = self._compute_target_joint_angles(self.target_pos)

        # 设置起始位置
        for i in range(6):
            self.data.qpos[i] = self.start_joint_angles[i]

        # 设置目标位置在场景中
        self.data.site_xpos[1] = self.target_pos  # site 1是目标位置

        # 执行前向动力学
        mj.mj_forward(self.model, self.data)

        # 重置内部状态
        self.current_step = 0
        self.prev_joint_angles = self.start_joint_angles.copy()
        self.prev_joint_velocities = np.zeros(6)
        self.joint_history = [self.start_joint_angles.copy()]

        return self._get_state()

    def _sample_random_joint_angles(self) -> np.ndarray:
        """采样随机关节角度"""
        angles = []
        for i in range(6):
            low, high = self.joint_limits[i]
            angles.append(np.random.uniform(low * 0.5, high * 0.5))  # 使用较小的范围
        return np.array(angles)

    def _sample_random_target_pos(self) -> np.ndarray:
        """采样随机目标位置"""
        x = np.random.uniform(self.target_pos_range['x'][0], self.target_pos_range['x'][1])
        y = np.random.uniform(self.target_pos_range['y'][0], self.target_pos_range['y'][1])
        z = np.random.uniform(self.target_pos_range['z'][0], self.target_pos_range['z'][1])
        return np.array([x, y, z])

    # RL-PID混合控制：不再需要逆运动学计算
    # def _compute_target_joint_angles(self, target_pos: np.ndarray) -> np.ndarray:
        """
        计算目标关节角度（使用改进的逆运动学）

        Args:
            target_pos: 目��位置 [x, y, z] (米)

        Returns:
            target_angles: 目标关节角度 [6] (弧度)
        """
        # 预先检查目标位置是否在合理工作空间内
        x, y, z = target_pos
        distance_from_base = np.sqrt(x**2 + y**2)

        # UR10e的典型工作空间参数（保守估计）
        max_reach = 1.1  # 最大可达距离
        min_reach = 0.3  # 最小可达距离
        max_height = 0.7  # 最大高度
        min_height = 0.15  # 最小高度

        # 如果目标点明显超出工作空间，直接使用简化方法
        if (distance_from_base > max_reach or distance_from_base < min_reach or
            z < min_height or z > max_height):
            # 使用简化几何方法，减少警告输出
            return self._fallback_ik(target_pos)

        # 使用改进的运动学解算器
        solutions = self.kinematics.inverse_kinematics_position(target_pos)

        if not solutions:
            # 如果无解，使用简化的几何方法作为备用
            return self._fallback_ik(target_pos)

        # 选择与当前关节角度最接近的解
        current_angles = self.data.qpos[:6].copy()
        best_solution = self.kinematics.select_best_solution(solutions, current_angles)

        if best_solution is None:
            # 如果所有解都无效，使用简化方法
            return self._fallback_ik(target_pos)
        print("⚠️使用_compute_target_joint_angles")
        return np.array(best_solution)

    def _fallback_ik(self, target_pos: np.ndarray) -> np.ndarray:
        """
        备用的简化逆运动学方法（仅当精确IK失败时使用）

        Args:
            target_pos: 目标位置 [x, y, z]

        Returns:
            target_angles: 目标关节角度 [6]
        """
        x, y, z = target_pos

        # 第一关节：绕Z轴旋转
        theta1 = np.arctan2(y, x)

        # 到目标点的距离（在XY平面）
        r = np.sqrt(x**2 + y**2)

        # UR10e的连杆长度
        l1 = 0.612  # shoulder到elbow
        l2 = 0.572  # elbow到wrist1

        # 计算第二和第三关节
        d = np.sqrt(r**2 + (z - 0.1807)**2)  # 减去基座高度

        # 检查是否在工作空间内
        if d > l1 + l2:
            d = l1 + l2 * 0.9

        # 使用余弦定理计算关节角度
        cos_theta3 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta3 = np.clip(cos_theta3, -1, 1)

        theta3 = np.arccos(cos_theta3)

        # 计算theta2
        alpha = np.arctan2(z - 0.1807, r)
        beta = np.arctan2(l2 * np.sin(theta3), l1 + l2 * np.cos(theta3))
        theta2 = alpha - beta

        # 腕部关节（保持末端垂直）
        theta4 = -theta2 - theta3
        theta5 = 0.0
        theta6 = 0.0
        print("⚠️使用_fallback_ik")

        return np.array([theta1, theta2, theta3, theta4, theta5, theta6])

    def _get_state(self) -> np.ndarray:
        """
        获取当前状态（RL-PID混合控制，16维）

        Returns:
            state: 16维状态向量
                [current_angles(6), current_velocities(6),
                 position_error(3), distance_to_target(1)]
        """
        try:
            # 获取当前关节角度
            current_joint_angles = self.data.qpos[:6].copy()
            if np.isnan(current_joint_angles).any():
                current_joint_angles = np.zeros(6)

            # 获取当前末端位置
            current_end_pos = self.data.site_xpos[0].copy()
            if np.isnan(current_end_pos).any():
                current_end_pos = np.zeros(3)

            # 获取当前关节速度
            current_joint_velocities = self.data.qvel[:6].copy()
            if np.isnan(current_joint_velocities).any():
                current_joint_velocities = np.zeros(6)

            # 验证目标位置
            if self.target_pos is None or np.isnan(self.target_pos).any():
                target_pos = np.zeros(3)
            else:
                target_pos = self.target_pos

            # 计算位置误差（3维）
            position_error = target_pos - current_end_pos

            # 计算欧氏距离
            distance_to_target = np.linalg.norm(position_error)
            if np.isnan(distance_to_target) or np.isinf(distance_to_target):
                distance_to_target = 1.0

            # 构建16维状态向量（归一化）
            normalized_state = np.concatenate([
                self._normalize_joint_positions(current_joint_angles),    # 关节位置归一化 [-1,1]
                self._normalize_joint_velocities(current_joint_velocities), # 关节速度归一化 [-1,1]
                self._normalize_position_error(position_error),          # 位置误差归一化 [-1,1]
                [self._normalize_distance(distance_to_target)]            # 距离归一化 [0,1] (包装为数组)
            ])

            # 数值稳定性检查
            if np.isnan(normalized_state).any() or np.isinf(normalized_state).any():
                print("⚠️  归一化状态包含NaN或Inf值，使用零向量")
                return np.zeros(16)

            return normalized_state

        except Exception as e:
            print(f"❌ 状态计算错误: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(16)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步动作（RL-PID混合控制）

        Args:
            action: 3维PID调度向量
                   [kp_scale, kd_scale, ki_enable]
                   - kp_scale: P增益缩放因子 [-0.5, 0.5]
                   - kd_scale: D增益缩放因子 [-0.5, 0.5]
                   - ki_enable: 积分控制启用标志 [0, 1]

        Returns:
            next_state: 下一状态（16维）
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 裁剪动作到允许范围
        action = np.clip(action, -self.action_bound, self.action_bound)

        # 解析PID调度动作
        kp_scale = action[0]  # P增益缩放因子
        kd_scale = action[1]  # D增益缩放因子
        ki_enable = max(0, action[2])  # 积分控制启用（确保非负）

        # 获取当前状态（用于奖励计算）
        current_joint_angles = self.data.qpos[:6].copy()
        current_joint_velocities = self.data.qvel[:6].copy()
        current_end_pos = self.data.site_xpos[0].copy()

        # 计算奖励（使用混合控制奖励函数）
        reward, reward_components = self._compute_hybrid_control_reward(
            action, current_joint_angles, current_joint_velocities,
            current_end_pos, self.target_pos
        )

        # RL调度PID参数的控制器
        control_torques = self._rl_pid_control(
            current_joint_angles, current_joint_velocities,
            kp_scale, kd_scale, ki_enable
        )
        self.data.ctrl[:6] = control_torques

        # 执行仿真步
        mj.mj_step(self.model, self.data)

        # 更新历史数据
        self.prev_joint_angles = current_joint_angles.copy()
        self.prev_joint_velocities = current_joint_velocities.copy()
        self.joint_history.append(self.data.qpos[:6].copy())
        if len(self.joint_history) > 3:
            self.joint_history.pop(0)

        # 渲染
        if self.enable_rendering:
            self._render()

        # 获取下一状态
        next_state = self._get_state()

        # 计算结束条件
        pos_error = np.linalg.norm(self.data.site_xpos[0] - self.target_pos)
        done = self._should_terminate(pos_error)

        # 更新步数
        self.current_step += 1

        # 构建信息字典
        info = {
            'pos_error': pos_error,
            'success': pos_error < self.reward_config['accuracy']['threshold'],
            'step': self.current_step,
            'reward_components': reward_components
        }

        return next_state, reward, done, info

    def _compute_multi_objective_reward(self, action: np.ndarray, joint_angles: np.ndarray,
                                       joint_velocities: np.ndarray, end_pos: np.ndarray,
                                       target_pos: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        计算多目标奖励函数（基于论文设计）

        r(s_t, a_t) = r_a^t + r_s^t + r_e^t + r_ex^t

        Args:
            action: 关节角速度（6维）
            joint_angles: 当前关节角度（6维）
            joint_velocities: 当前关节速度（6维）
            end_pos: 末端位置（3维）
            target_pos: 目标位置（3维）

        Returns:
            total_reward: 总奖励
            components: 各奖励分量
        """
        # 1. 精度奖励 r_a^t = -w_a * exp(σ_a * f_a(θ^t))
        # f_a(θ^t) = ||p_d - p||^2 是位置误差的平方（论文设计）
        pos_error = np.linalg.norm(target_pos - end_pos)  # 位置误差
        f_a_theta = pos_error ** 2  # f_a(θ^t) = ||p_d - p||^2

        # 使用指数惩罚：误差小时惩罚温和，误差大时惩罚急剧增加
        accuracy_reward = -self.reward_config['accuracy']['weight'] * np.exp(
            self.reward_config['accuracy']['sigma'] * f_a_theta
        )

        # 2. 平滑性奖励 r_s^t = -w_s * sum(λ_v * q̇_k^2 + λ_acc * q̈_k^2)
        velocity_penalty = np.sum(joint_velocities ** 2) * self.reward_config['smoothness']['lambda_vel']

        # 计算加速度（如果有历史数据）
        if len(self.joint_history) >= 2:
            acc = (self.joint_history[-1] - self.joint_history[-2]) / 0.01  # 近似加速度
            acceleration_penalty = np.sum(acc ** 2) * self.reward_config['smoothness']['lambda_acc']
        else:
            acceleration_penalty = 0.0

        smoothness_reward = -self.reward_config['smoothness']['weight'] * (velocity_penalty + acceleration_penalty)

        # 3. 能量消耗奖励 r_e^t = -w_e * sum((Δθ_k * τ_k)^2)
        # 这里使用动作幅度与关节位置的乘积作为功率的近似
        torque_approx = np.abs(joint_angles)  # 简化的扭矩近似
        power_approx = action * torque_approx
        energy_penalty = np.sum(power_approx ** 2)
        energy_reward = -self.reward_config['energy']['weight'] * energy_penalty

        # 4. 额外奖励 r_ex^t（稀疏奖励）
        if pos_error < self.reward_config['accuracy']['threshold']:
            extra_reward = self.reward_config['extra']['success_reward']
        else:
            extra_reward = self.reward_config['extra']['success_reward'] / (1.0 + 10.0 * pos_error)

        # 总奖励
        total_reward = accuracy_reward + smoothness_reward + energy_reward + extra_reward

        # 奖励分量
        components = {
            'accuracy': accuracy_reward,
            'smoothness': smoothness_reward,
            'energy': energy_reward,
            'extra': extra_reward,
            'pos_error': pos_error
        }

        return total_reward, components

    def _should_terminate(self, pos_error: float) -> bool:
        """
        判断是否应该终止episode（包含衰减回合机制）

        Args:
            pos_error: 当前位置误差

        Returns:
            should_terminate: 是否终止
        """
        # 如果已经成功到达目标
        if pos_error < self.reward_config['accuracy']['threshold']:
            return True

        # 如果超过当前最大步数
        if self.current_step >= self.current_max_steps:
            return True

        return False

    def _render(self):
        """渲染场景"""
        if self.enable_rendering and self.viewer is not None:
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.viewer)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # 更新场景
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                              mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # 交换缓冲区
            glfw.swap_buffers(self.viewer)
            glfw.poll_events()

    def update_decay_episode_mechanism(self, episode_reward: float, final_pos_error: float):
        """
        更新衰减回合机制（论文创新点）

        Args:
            episode_reward: episode总奖励
            final_pos_error: 最终位置误差
        """
        if not self.decaying_episode_enabled:
            return

        # 判断是否成功
        is_success = final_pos_error < self.reward_config['accuracy']['threshold']

        # 记录历史
        self.success_history.append(is_success)

        # 保持窗口大小
        if len(self.success_history) > self.decay_window:
            self.success_history.pop(0)

        # 检查是否需要衰减
        if len(self.success_history) >= self.decay_window // 2:
            recent_success_rate = sum(self.success_history[-(self.decay_window // 2):]) / (self.decay_window // 2)

            # 如果最近成功率足够高
            if recent_success_rate >= self.success_threshold:
                # 减少最大步数
                new_max_steps = max(self.min_max_steps, self.current_max_steps - 50)

                if new_max_steps < self.current_max_steps:
                    print(f"[衰减回合] 成功率 {recent_success_rate:.2f} >= {self.success_threshold}")
                    print(f"[衰减回合] 最大步数: {self.current_max_steps} -> {new_max_steps}")
                    self.current_max_steps = new_max_steps

                    # 清空历史以重新评估
                    self.success_history = []

    def get_decay_stats(self) -> Dict[str, Any]:
        """获取衰减回合统计信息"""
        return {
            'current_max_steps': self.current_max_steps,
            'success_rate': sum(self.success_history) / len(self.success_history) if self.success_history else 0.0,
            'decay_enabled': self.decaying_episode_enabled
        }

    def close(self):
        """关闭环境"""
        if self.enable_rendering and self.viewer is not None:
            glfw.terminate()

    def get_state_space_size(self) -> int:
        """获取状态空间大小"""
        return 25

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return 6