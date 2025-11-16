"""
UR10e PPO 多目标最优轨迹规划工具函数

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》
提供训练、评估和可视化的辅助功能
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import math


class ValueNormalization(nn.Module):
    """
    Value Function Normalization

    用于稳定Critic训练的值函数归一化技术
    基于Isaac Gym实现，提供在线更新和归一化功能
    """
    def __init__(self, beta: float = 0.995, epsilon: float = 1e-8, clip_range: float = 10.0):
        super().__init__()
        self.beta = beta          # 指数移动平均系数
        self.epsilon = epsilon      # 数值稳定性参数
        self.clip_range = clip_range # 归一化值裁剪范围

        # 可学习的参数
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('var', torch.ones(1))
        self.register_buffer('count', torch.zeros(1))

    def update(self, values: torch.Tensor):
        """
        更新归一化统计量（在线EMA更新）

        Args:
            values: [batch_size, 1] 或 [batch_size] 价值函数值
        """
        values = values.view(-1, 1) if values.dim() == 1 else values

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = values.numel()

        # 在线更新均值和方差
        self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
        self.var = self.beta * self.var + (1 - self.beta) * batch_var
        self.count += batch_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        归一化值函数

        Args:
            values: 输入值

        Returns:
            normalized_values: 归一化后的值
        """
        values = values.view(-1, 1) if values.dim() == 1 else values

        std = torch.sqrt(self.var + self.epsilon)
        normalized = (values - self.mean) / std
        return torch.clamp(normalized, -self.clip_range, self.clip_range).squeeze(-1)

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """
        反归一化值函数

        Args:
            normalized_values: 归一化后的值

        Returns:
            denormalized_values: 原始尺度的值
        """
        normalized_values = normalized_values.view(-1, 1) if normalized_values.dim() == 1 else normalized_values

        std = torch.sqrt(self.var + self.epsilon)
        denormalized = normalized_values * std + self.mean
        return denormalized.squeeze(-1)


class GAE:
    """
    Generalized Advantage Estimation (GAE)

    计算优势函数和回报的稳定方法，支持自适应折扣因子
    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95,
                 device: torch.device = None, use_adaptive_gamma: bool = False,
                 eta_min: float = 0.6, eta_max: float = 0.99):
        self.gamma = gamma              # 折扣因子
        self.lam = lam                  # GAE的λ参数
        self.device = device or torch.device('cpu')
        self.use_adaptive_gamma = use_adaptive_gamma
        self.eta_min = eta_min
        self.eta_max = eta_max

    def compute_adaptive_gamma(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        计算自适应折扣因子（可选功能）

        Args:
            action_probs: [T, N] 动作概率（策略质量指标）

        Returns:
            adaptive_gamma: [T, N] 自适应折扣因子
        """
        # 将动作概率映射到折扣因子范围 [eta_min, eta_max]
        # 概率越高（策略质量越好），折扣因子越大
        adaptive_gamma = self.eta_min + (self.eta_max - self.eta_min) * action_probs
        return torch.clamp(adaptive_gamma, self.eta_min, self.eta_max)

    def __call__(self, rewards: torch.Tensor, dones: torch.Tensor,
                 values: torch.Tensor, next_values: torch.Tensor,
                 action_probs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势函数和回报

        Args:
            rewards: [T, N] 奖励序列
            dones: [T, N] 结束标志
            values: [T, N] 价值函数序列
            next_values: [T, N] 下一状态价值函数
            action_probs: [T, N] 动作概率（用于自适应折扣因子）

        Returns:
            advantages: [T, N] 优势函数
            returns: [T, N] 回报
        """
        T, N = rewards.shape

        # 确保所有输入张量在正确的设备上
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)
        next_values = next_values.to(self.device)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # 计算自适应折扣因子
        if self.use_adaptive_gamma and action_probs is not None:
            action_probs = action_probs.to(self.device)
            gamma_t = self.compute_adaptive_gamma(action_probs)
        else:
            gamma_t = torch.full_like(rewards, self.gamma)

        # GAE计算
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            # 计算TD误差
            delta = rewards[t] + gamma_t[t] * next_value * (1 - dones[t]) - values[t]

            # GAE更新
            gae = delta + gamma_t[t] * self.lam * (1 - dones[t]) * gae

            # 保存结果
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保实验可复现性

    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 随机种子已设置为: {seed}")


def compute_trajectory_smoothness(trajectory: np.ndarray) -> float:
    """
    计算轨迹平滑度指标

    Args:
        trajectory: [T, 6] 关节角度或位置序列

    Returns:
        smoothness: 平滑度指标（值越小越平滑）
    """
    if len(trajectory) < 3:
        return 0.0

    # 计算一阶差分（速度）
    velocity = np.diff(trajectory, axis=0)

    # 计算二阶差分（加速度）
    acceleration = np.diff(velocity, axis=0)

    # 平滑度指标：加速度的L2范数的平均值
    if len(acceleration) > 0:
        smoothness = np.mean(np.linalg.norm(acceleration, axis=1))
    else:
        smoothness = 0.0

    return smoothness


def compute_trajectory_metrics(trajectory_data: List[np.ndarray],
                             target_positions: List[np.ndarray],
                             success_threshold: float = 0.005) -> Dict[str, float]:
    """
    计算轨迹质量指标

    Args:
        trajectory_data: 轨迹数据列表
        target_positions: 目标位置列表
        success_threshold: 成功阈值

    Returns:
        metrics: 指标字典
    """
    if not trajectory_data:
        return {}

    metrics = {
        'avg_trajectory_length': 0.0,
        'avg_smoothness': 0.0,
        'success_rate': 0.0,
        'avg_final_error': 0.0,
        'trajectory_consistency': 0.0
    }

    # 计算轨迹长度
    lengths = [len(traj) for traj in trajectory_data]
    metrics['avg_trajectory_length'] = np.mean(lengths)

    # 计算平滑度
    smoothness_values = [compute_trajectory_smoothness(traj) for traj in trajectory_data]
    metrics['avg_smoothness'] = np.mean(smoothness_values)

    # 计算成功率和最终误差
    final_errors = []
    successful_count = 0

    for i, (traj, target) in enumerate(zip(trajectory_data, target_positions)):
        if len(traj) > 0:
            # 假设轨迹的最后一列是末端位置
            if traj.shape[1] >= 3:
                final_pos = traj[-1, :3]
                final_error = np.linalg.norm(final_pos - target)
                final_errors.append(final_error)

                if final_error < success_threshold:
                    successful_count += 1

    if final_errors:
        metrics['avg_final_error'] = np.mean(final_errors)
        metrics['success_rate'] = successful_count / len(final_errors)

    # 计算轨迹一致性（轨迹之间的相似度）
    if len(trajectory_data) > 1:
        # 简化实现：计算轨迹长度的标准差
        metrics['trajectory_consistency'] = 1.0 / (1.0 + np.std(lengths))

    return metrics


def save_training_data(episode_data: List[Dict[str, Any]],
                      filepath: str = "csv_output/training_data.csv"):
    """
    保存训练数据到CSV文件

    Args:
        episode_data: 训练数据列表
        filepath: 保存路径
    """
    df = pd.DataFrame(episode_data)
    df.to_csv(filepath, index=False)
    print(f"📊 训练数据已保存到: {filepath}")


def plot_training_curves(training_stats: Dict[str, List],
                        config: Optional[Dict[str, Any]] = None,
                        save_path: str = None,
                        show_plots: bool = False):
    """
    绘制训练曲线

    Args:
        training_stats: 训练统计数据
        config: 配置字典
        save_path: 保存路径
        show_plots: 是否显示图像
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('UR10e PPO Training Progress', fontsize=16)

    # Episode奖励
    if 'episode_rewards' in training_stats and training_stats['episode_rewards']:
        axes[0, 0].plot(training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # 添加平滑曲线
        window_size = min(100, len(training_stats['episode_rewards']))
        if window_size > 0:
            smooth_rewards = pd.Series(training_stats['episode_rewards']).rolling(window=window_size).mean()
            axes[0, 0].plot(smooth_rewards, label=f'Smoothed ({window_size})', alpha=0.7)
            axes[0, 0].legend()

    # Episode长度
    if 'episode_lengths' in training_stats and training_stats['episode_lengths']:
        axes[0, 1].plot(training_stats['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)

    # 位置误差
    if 'position_errors' in training_stats and training_stats['position_errors']:
        axes[0, 2].plot(training_stats['position_errors'])
        axes[0, 2].set_title('Position Errors')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Error (m)')
        axes[0, 2].grid(True)

        # 添加成功阈值线
        success_threshold = config.get('reward', {}).get('accuracy', {}).get('threshold', 0.005) if config else 0.005
        axes[0, 2].axhline(y=success_threshold, color='r', linestyle='--', label='Success Threshold')
        axes[0, 2].legend()

    # 成功率
    if 'success_rates' in training_stats and training_stats['success_rates']:
        axes[1, 0].plot(training_stats['success_rates'])
        axes[1, 0].set_title('Success Rate (100-episode window)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])

    # 衰减回合统计
    if 'decay_stats' in training_stats and training_stats['decay_stats']:
        decay_steps = [stats.get('current_max_steps', 1000) for stats in training_stats['decay_stats']]
        axes[1, 1].plot(decay_steps)
        axes[1, 1].set_title('Decaying Episode Max Steps')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Max Steps')
        axes[1, 1].grid(True)

    # 奖励分量分析
    if 'reward_components' in training_stats and training_stats['reward_components']:
        # 提取奖励分量
        accuracy_rewards = []
        smoothness_rewards = []
        energy_rewards = []

        for components in training_stats['reward_components']:
            if isinstance(components, dict):
                accuracy_rewards.append(components.get('accuracy', 0))
                smoothness_rewards.append(components.get('smoothness', 0))
                energy_rewards.append(components.get('energy', 0))

        if accuracy_rewards:
            axes[1, 2].plot(accuracy_rewards, label='Accuracy', alpha=0.7)
        if smoothness_rewards:
            axes[1, 2].plot(smoothness_rewards, label='Smoothness', alpha=0.7)
        if energy_rewards:
            axes[1, 2].plot(energy_rewards, label='Energy', alpha=0.7)

        axes[1, 2].set_title('Reward Components')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].grid(True)
        axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线已保存到: {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def visualize_trajectory(trajectory_data: np.ndarray,
                        target_position: np.ndarray,
                        save_path: str = None,
                        show_plot: bool = False):
    """
    可视化单个轨迹

    Args:
        trajectory_data: [T, 6] 轨迹数据
        target_position: [3] 目标位置
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if trajectory_data.shape[1] >= 3:
        ax1.plot(trajectory_data[:, 0], trajectory_data[:, 1], trajectory_data[:, 2], 'b-', label='Trajectory')
        ax1.scatter(target_position[0], target_position[1], target_position[2],
                   c='r', s=100, marker='*', label='Target')
        ax1.scatter(trajectory_data[0, 0], trajectory_data[0, 1], trajectory_data[0, 2],
                   c='g', s=50, marker='o', label='Start')
        ax1.scatter(trajectory_data[-1, 0], trajectory_data[-1, 1], trajectory_data[-1, 2],
                   c='orange', s=50, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # 关节角度
    for i in range(6):
        ax2 = fig.add_subplot(2, 3, i + 2)
        ax2.plot(trajectory_data[:, i])
        ax2.set_title(f'Joint {i + 1} Angle')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Angle (rad)')
        ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 轨迹可视化已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_experiment_directory(base_dir: str = "./experiments") -> str:
    """
    创建实验目录

    Args:
        base_dir: 基础目录

    Returns:
        experiment_dir: 实验目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"ur10e_ppo_{timestamp}")

    # 创建子目录
    subdirs = [
        "checkpoints",      # 模型检查点
        "logs",            # 日志文件
        "csv_output",      # CSV数据
        "plots",           # 训练曲线
        "trajectories",    # 轨迹可视化
        "config",          # 配置文件
        "models"           # 最终模型
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    print(f"📁 实验目录已创建: {experiment_dir}")
    return experiment_dir


def save_experiment_config(config: Dict[str, Any], experiment_dir: str):
    """
    保存实验配置

    Args:
        config: 配置字典
        experiment_dir: 实验目录
    """
    config_path = os.path.join(experiment_dir, "config", "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 同时保存为JSON格式
    json_path = os.path.join(experiment_dir, "config", "config.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"⚙️  实验配置已保存")


def compute_success_metrics(position_errors: List[float],
                           threshold: float = 0.005) -> Dict[str, float]:
    """
    计算成功指标

    Args:
        position_errors: 位置误差列表
        threshold: 成功阈值

    Returns:
        metrics: 成功指标字典
    """
    if not position_errors:
        return {}

    success_count = sum(1 for error in position_errors if error < threshold)
    total_count = len(position_errors)

    metrics = {
        'success_rate': success_count / total_count,
        'total_episodes': total_count,
        'successful_episodes': success_count,
        'mean_error': np.mean(position_errors),
        'std_error': np.std(position_errors),
        'median_error': np.median(position_errors),
        'min_error': np.min(position_errors),
        'max_error': np.max(position_errors)
    }

    return metrics


def generate_training_report(training_stats: Dict[str, List],
                           config: Dict[str, Any],
                           experiment_dir: str):
    """
    生成训练报告

    Args:
        training_stats: 训练统计数据
        config: 配置字典
        experiment_dir: 实验目录
    """
    report_path = os.path.join(experiment_dir, "training_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("UR10e PPO 多目标最优轨迹规划训练报告\n")
        f.write("=" * 50 + "\n\n")

        # 配置信息
        f.write("📋 配置信息:\n")
        f.write(f"最大训练轮数: {config['train']['max_episodes']}\n")
        f.write(f"状态空间: 25维\n")
        f.write(f"动作空间: 6维\n")
        f.write(f"最大步数: {config['env']['max_steps']}\n")
        f.write(f"衰减回合机制: {'启用' if config['decay_episode']['enabled'] else '禁用'}\n")
        f.write(f"成功率阈值: {config['decay_episode']['success_threshold']}\n\n")

        # 训练统计
        if 'episode_rewards' in training_stats and training_stats['episode_rewards']:
            f.write("📊 训练统计:\n")
            f.write(f"总训练轮数: {len(training_stats['episode_rewards'])}\n")
            f.write(f"最终平均奖励: {np.mean(training_stats['episode_rewards'][-100:]):.2f}\n")
            f.write(f"最终成功率: {training_stats['success_rates'][-1]:.2%}\n")
            f.write(f"最终平均位置误差: {np.mean(training_stats['position_errors'][-100:]):.4f}m\n")

            # 成功指标
            if 'position_errors' in training_stats:
                success_metrics = compute_success_metrics(training_stats['position_errors'])
                f.write(f"\n🎯 成功指标:\n")
                for key, value in success_metrics.items():
                    if key == 'success_rate':
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")

    print(f"📝 训练报告已保存到: {report_path}")


class RewardNormalizer:
    """
    奖励归一化器

    用于稳定PPO训练的奖励归一化技术，支持在线更新和多种归一化策略
    """

    def __init__(self,
                 gamma: float = 0.99,
                 clip_range: float = 5.0,
                 epsilon: float = 1e-8,
                 normalize_method: str = 'running_stats',
                 warmup_steps: int = 100,
                 history_size: int = 10000):
        """
        初始化奖励归一化器

        Args:
            gamma: 折扣因子，用于计算折扣奖励统计
            clip_range: 归一化值裁剪范围
            epsilon: 数值稳定性参数
            normalize_method: 归一化方法 ['running_stats', 'batch_stats', 'rank']
            warmup_steps: 预热步数，初期不进行归一化
            history_size: 奖励历史记录大小
        """
        self.gamma = gamma
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.normalize_method = normalize_method
        self.warmup_steps = warmup_steps
        self.history_size = history_size

        # 运行时统计量
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 0
        self.beta = 0.99  # 指数移动平均系数

        # 奖励历史
        self.reward_history = []
        self.discounted_reward_history = []

        # 批次统计
        self.batch_rewards = []

    def update(self, reward: float, done: bool = False):
        """
        更新归一化器统计量

        Args:
            reward: 当前奖励值
            done: 是否回合结束
        """
        self.reward_history.append(reward)
        self.running_count += 1

        # 指数移动平均更新
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward
        delta = reward - self.running_mean
        self.running_var = self.beta * self.running_var + (1 - self.beta) * delta * delta

        # 维护历史记录在合理范围内
        if len(self.reward_history) > self.history_size:
            self.reward_history = self.reward_history[-self.history_size//2:]

        # 回合结束时计算折扣奖励统计
        if done and len(self.reward_history) > 1:
            self._update_discounted_stats()

    def _update_discounted_stats(self):
        """更新折扣奖励统计"""
        if not self.reward_history:
            return

        # 计算最近一个episode的折扣奖励
        discounted_rewards = []
        reward_sum = 0.0
        for reward in reversed(self.reward_history):
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        self.discounted_reward_history.extend(discounted_rewards)

        # 维护折扣奖励历史
        if len(self.discounted_reward_history) > self.history_size:
            self.discounted_reward_history = self.discounted_reward_history[-self.history_size//2:]

    def normalize(self, reward: float) -> float:
        """
        归一化单个奖励

        Args:
            reward: 原始奖励值

        Returns:
            normalized_reward: 归一化后的奖励值
        """
        if self.running_count < self.warmup_steps:
            return reward  # 预热期不归一化

        if self.normalize_method == 'running_stats':
            return self._normalize_running_stats(reward)
        elif self.normalize_method == 'batch_stats':
            return self._normalize_batch_stats(reward)
        elif self.normalize_method == 'rank':
            return self._normalize_rank(reward)
        else:
            return reward

    def _normalize_running_stats(self, reward: float) -> float:
        """使用运行统计量归一化"""
        std = np.sqrt(self.running_var + self.epsilon)
        normalized = (reward - self.running_mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _normalize_batch_stats(self, reward: float) -> float:
        """使用批次统计量归一化"""
        if len(self.reward_history) < 10:
            return reward

        # 使用最近的奖励作为批次
        recent_rewards = self.reward_history[-min(100, len(self.reward_history)):]
        batch_mean = np.mean(recent_rewards)
        batch_std = np.std(recent_rewards) + self.epsilon

        normalized = (reward - batch_mean) / batch_std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _normalize_rank(self, reward: float) -> float:
        """使用秩归一化（均匀分布）"""
        if len(self.reward_history) < 10:
            return reward

        # 计算当前奖励在历史中的百分位
        count_smaller = sum(1 for r in self.reward_history if r < reward)
        percentile = count_smaller / len(self.reward_history)

        # 映射到[-1, 1]范围
        normalized = 2 * percentile - 1
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def normalize_batch(self, rewards: np.ndarray) -> np.ndarray:
        """
        批量归一化奖励

        Args:
            rewards: [batch_size] 奖励数组

        Returns:
            normalized_rewards: 归一化后的奖励数组
        """
        if self.running_count < self.warmup_steps:
            return rewards

        normalized_rewards = np.array([self.normalize(r) for r in rewards])
        return normalized_rewards

    def get_stats(self) -> dict:
        """获取归一化器统计信息"""
        return {
            'method': self.normalize_method,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'running_std': np.sqrt(self.running_var + self.epsilon),
            'count': self.running_count,
            'recent_mean': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'recent_std': np.std(self.reward_history[-100:]) if len(self.reward_history) > 1 else 0.0,
            'history_size': len(self.reward_history),
            'warmup_progress': min(1.0, self.running_count / self.warmup_steps)
        }

    def reset(self):
        """重置归一化器（保留学习到的统计量）"""
        self.reward_history = []
        self.batch_rewards = []

    def full_reset(self):
        """完全重置归一化器"""
        self.reward_history = []
        self.discounted_reward_history = []
        self.batch_rewards = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 0


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        # 打印到控制台
        print(log_message)

    def log_experiment_start(self, config: Dict[str, Any]):
        """记录实验开始"""
        self.log("🚀 开始UR10e PPO训练实验")
        self.log(f"📋 配置: {config}")
        self.log(f"🎯 状态空间: 25维, 动作空间: 6维")
        self.log(f"🔧 衰减回合机制: {'启用' if config['decay_episode']['enabled'] else '禁用'}")

    def log_experiment_end(self, final_stats: Dict[str, Any]):
        """记录实验结束"""
        self.log("🎉 训练完成！")
        self.log(f"📊 最终统计: {final_stats}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的有效性

    Args:
        config: 配置字典

    Returns:
        is_valid: 配置是否有效
    """
    try:
        # 检查必需的配置项
        required_sections = ['env', 'ppo', 'train', 'reward']
        for section in required_sections:
            if section not in config:
                print(f"❌ 配置缺少必需部分: {section}")
                return False

        # 检查环境配置
        if 'xml_path' not in config['env']:
            print("❌ 缺少XML模型路径")
            return False

        # 检查PPO配置
        ppo_required = ['lr_actor', 'lr_critic', 'clip_eps', 'gamma']
        for key in ppo_required:
            if key not in config['ppo']:
                print(f"❌ PPO配置缺少必需项: {key}")
                return False

        # 检查奖励配置
        reward_required = ['accuracy', 'smoothness', 'energy']
        for key in reward_required:
            if key not in config['reward']:
                print(f"❌ 奖励配置缺少必需项: {key}")
                return False

        print("✅ 配置文件验证通过")
        return True

    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False