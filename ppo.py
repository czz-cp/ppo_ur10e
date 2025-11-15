"""
PPO算法实现

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》
针对UR10e轨迹规划任务的PPO算法实现

���心特性：
1. PPO Clip机制确保训练稳定性
2. 25维状态空间 + 6维动作空间
3. 高斯分布策略头（适合连续控制）
4. 支持多目标奖励函数
5. 使用utils.py中的ValueNormalization和GAE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

# 从utils导入工具类
try:
    from utils import ValueNormalization, GAE
except ImportError:
    # 如果导入失败，提供本地实现
    print("⚠️  无法从utils导入ValueNormalization和GAE，使用本地实现")

    class ValueNormalization(nn.Module):
        """本地ValueNormalization实现"""
        def __init__(self, beta: float = 0.995, epsilon: float = 1e-8, clip_range: float = 10.0):
            super().__init__()
            self.beta = beta
            self.epsilon = epsilon
            self.clip_range = clip_range
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('var', torch.ones(1))
            self.register_buffer('count', torch.zeros(1))

        def update(self, values: torch.Tensor):
            batch_mean = values.mean()
            batch_var = values.var(unbiased=False)
            batch_count = values.numel()
            self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            self.var = self.beta * self.var + (1 - self.beta) * batch_var
            self.count += batch_count

        def normalize(self, values: torch.Tensor) -> torch.Tensor:
            std = torch.sqrt(self.var + self.epsilon)
            normalized = (values - self.mean) / std
            return torch.clamp(normalized, -self.clip_range, self.clip_range)

        def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
            std = torch.sqrt(self.var + self.epsilon)
            return normalized_values * std + self.mean

    class GAE:
        """本地GAE实现"""
        def __init__(self, gamma: float = 0.99, lam: float = 0.95):
            self.gamma = gamma
            self.lam = lam

        def __call__(self, rewards: torch.Tensor, dones: torch.Tensor,
                     values: torch.Tensor, next_values: torch.Tensor,
                     action_probs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            T, N = rewards.shape
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            gae = torch.zeros(N)

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = next_values[t]
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values[t]

            return advantages, returns


class ActorNetwork(nn.Module):
    """
    Actor网络（策略网络）

    基于Isaac训练模式设计的深度网络架构
    专为UR10e轨迹规划任务优化
    """
    def __init__(self, state_dim: int = 25, action_dim: int = 6, hidden_dim: int = 256):
        super().__init__()

        # 特征提取网络 - 深层架构
        self.feature_extractor = nn.Sequential(
            # 第一层：状态→特征
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.1),  # LeakyReLU比Tanh更适合梯度流
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm稳定训练

            # 第二层：特征扩展
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),

            # 第三层：特征压缩
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim // 2),
        )

        # 策略头 - 输出均值
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, action_dim)
        )

        # 策略头 - 输出标准差
        self.policy_std = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, action_dim)
        )

        # 可学习的log_std缩放参数
        self.log_std_min = -2.0
        self.log_std_max = 0.5

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重 - 基于Isaac训练的模式"""
        def init_(module):
            if isinstance(module, nn.Linear):
                # 正交初始化 + 0.01缩放（Isaac验证的稳定模式）
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: [batch_size, state_dim] 状态

        Returns:
            mean: [batch_size, action_dim] 动作均值
            log_std: [batch_size, action_dim] 动作log标准差
        """
        # 基本数值检查
        if torch.isnan(state).any():
            print("⚠️  输入状态包含NaN值!")
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # 特征提取
        features = self.feature_extractor(state)

        # 分别计算均值和标准差
        mean = self.policy_mean(features)
        log_std = self.policy_std(features)

        # 限制log_std范围和数值稳定性
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        mean = torch.clamp(mean, -10.0, 10.0)  # 限制均值范围

        return mean, log_std

    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作

        Args:
            state: [batch_size, state_dim] 状态
            deterministic: 是否使用确定性策略

        Returns:
            action: [batch_size, action_dim] 采样动作
            log_prob: [batch_size] 动作的对数概率
        """
        mean, log_std = self.forward(state)

        # 检查网络输出
        if torch.isnan(mean).any():
            print("⚠️  mean包含NaN!")
            mean = torch.zeros_like(mean)
        if torch.isnan(log_std).any():
            print("⚠️  log_std包含NaN!")
            log_std = torch.zeros_like(log_std)

        std = torch.exp(log_std)

        # 检查std是否有效
        if torch.isnan(std).any() or torch.isinf(std).any():
            print("⚠️  std包含异常值!")
            std = torch.ones_like(std)

        if deterministic:
            # 确定性策略：直接使用均值
            action = torch.tanh(mean) * self.action_bound  # 缩放到动作范围
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            # 随机策略：从高斯分布采样
            try:
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.rsample()  # 使用重参数化技巧

                # 检查原始动作
                if torch.isnan(raw_action).any() or torch.isinf(raw_action).any():
                    print("⚠️  raw_action包含异常值!")
                    raw_action = torch.zeros_like(raw_action)

                # 使用tanh将动作限制到[-1, 1]
                action = torch.tanh(raw_action) * 0.0189  # 缩放到动作范围

                # 计算对数概率（考虑tanh变换的雅可比行列式）
                log_prob = dist.log_prob(raw_action).sum(dim=-1)

                # 检查log_prob
                if torch.isnan(log_prob).any():
                    print("⚠️  log_prob包含NaN!")
                    log_prob = torch.zeros_like(log_prob)

                # tanh变换的雅可比修正项
                tanh_term = torch.tanh(raw_action).pow(2)
                tanh_correction = torch.log(1.0 - tanh_term + 1e-6).sum(dim=-1)

                if torch.isnan(tanh_correction).any():
                    tanh_correction = torch.zeros_like(tanh_correction)

                log_prob = log_prob - tanh_correction

            except Exception as e:
                print(f"⚠️  动作采样出错: {e}")
                action = torch.zeros(state.shape[0], 6, device=state.device) * 0.0189
                log_prob = torch.zeros(state.shape[0], device=state.device)

        return action, log_prob

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作的概率和熵

        Args:
            state: [batch_size, state_dim] 状态
            action: [batch_size, action_dim] 动作

        Returns:
            log_prob: [batch_size] 动作的对数概率
            entropy: [batch_size] 策略熵
        """
        mean, log_std = self.forward(state)

        # 关键检查：检查mean和log_std是否有效
        if torch.isnan(mean).any():
            print("❌ evaluate_action: mean包含NaN!")
            print(f"mean范围: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
            print(f"state范围: [{state.min().item():.4f}, {state.max().item():.4f}]")
            mean = torch.zeros_like(mean)

        if torch.isnan(log_std).any():
            print("❌ evaluate_action: log_std包含NaN!")
            log_std = torch.zeros_like(log_std)

        std = torch.exp(log_std)

        # 检查std
        if torch.isnan(std).any() or torch.isinf(std).any():
            print("❌ evaluate_action: std包含异常值!")
            std = torch.ones_like(std)

        # 创建分布 - 添加异常处理
        try:
            dist = torch.distributions.Normal(mean, std)
        except Exception as e:
            print(f"❌ 创建Normal分布失败: {e}")
            print(f"mean统计: min={mean.min().item():.4f}, max={mean.max().item():.4f}, mean={mean.mean().item():.4f}")
            print(f"std统计: min={std.min().item():.4f}, max={std.max().item():.4f}, mean={std.mean().item():.4f}")
            # 使用安全的默认值
            mean_safe = torch.zeros_like(mean)
            std_safe = torch.ones_like(std)
            dist = torch.distributions.Normal(mean_safe, std_safe)

        # 将动作转换回原始空间（逆tanh变换）
        action_clamped = torch.clamp(action / 0.0189, -0.99, 0.99)  # 避免atanh的数值问题
        raw_action = torch.atanh(action_clamped)  # 反tanh变换

        # 计算对数概率
        try:
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            if torch.isnan(log_prob).any():
                print("❌ evaluate_action: log_prob计算后包含NaN!")
                log_prob = torch.zeros_like(log_prob)
        except Exception as e:
            print(f"❌ log_prob计算失败: {e}")
            log_prob = torch.zeros(state.shape[0], device=state.device)

        # tanh变换的雅可比修正项
        tanh_correction = torch.log(1.0 - action_clamped.pow(2) + 1e-6).sum(dim=-1)
        if torch.isnan(tanh_correction).any():
            tanh_correction = torch.zeros_like(tanh_correction)

        log_prob = log_prob - tanh_correction

        # 计算熵
        try:
            entropy = dist.entropy().sum(dim=-1)
            if torch.isnan(entropy).any():
                print("❌ evaluate_action: entropy包含NaN!")
                entropy = torch.zeros_like(entropy)
        except Exception as e:
            print(f"❌ entropy计算失败: {e}")
            entropy = torch.zeros(state.shape[0], device=state.device)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic网络（价值函数网络）

    基于Isaac训练模式设计的深度价值网络
    专为UR10e轨迹规划状态价值评估优化
    """
    def __init__(self, state_dim: int = 25, hidden_dim: int = 256):
        super().__init__()

        # 特征提取网络 - 与Actor共享架构思路
        self.feature_extractor = nn.Sequential(
            # 第一层：状态→特征
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),

            # 第二层：特征扩展
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),

            # 第三层：特征压缩
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim // 2),
        )

        # 价值头 - 深层架构
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 8, 1)  # 输出单一价值
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重 - 基于Isaac训练的模式"""
        def init_(module):
            if isinstance(module, nn.Linear):
                # 正交初始化 + 0.01缩放（Isaac验证的稳定模式）
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: [batch_size, state_dim] 状态

        Returns:
            value: [batch_size] 状态价值
        """
        # 基本数值检查
        if torch.isnan(state).any():
            print("⚠️  Critic输入状态包含NaN值!")
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # 特征提取
        features = self.feature_extractor(state)

        # 价值预测
        value = self.value_head(features).squeeze(-1)

        # 数值稳定性
        value = torch.clamp(value, -1000.0, 1000.0)  # 防止极端值

        return value


class PPO:
    """
    PPO算法实现

    Proximal Policy Optimization，专门针对UR10e轨迹规划任务优化
    """
    def __init__(self, state_dim: int = 25, action_dim: int = 6,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 clip_eps: float = 0.2, gamma: float = 0.99, gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, epochs: int = 10, batch_size: int = 64,
                 action_bound: float = 0.03):

        # 超参数
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.action_bound = action_bound

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 网络
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        # Value Normalization
        self.value_norm = ValueNormalization().to(self.device)

        # GAE
        self.gae = GAE(gamma=gamma, lam=gae_lambda, device=self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 使用HuberLoss
        self.critic_loss_fn = nn.HuberLoss(delta=10.0)

        # 训练统计
        self.update_count = 0

    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               dones: torch.Tensor, next_states: torch.Tensor) -> Dict[str, float]:
        """
        PPO更新

        Args:
            states: [T, N, state_dim] 状态序列
            actions: [T, N, action_dim] 动作序列
            rewards: [T, N] 奖励序列
            dones: [T, N] 结束标志
            next_states: [T, N, state_dim] 下一状态序列

        Returns:
            stats: 训练统计信息
        """
        # 重塑数据
        T, N = states.shape[:2]
        states_flat = states.reshape(-1, states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        rewards_flat = rewards.reshape(-1)
        dones_flat = dones.reshape(-1)
        next_states_flat = next_states.reshape(-1, next_states.shape[-1])

        # 计算价值函数
        with torch.no_grad():
            values_flat = self.critic(states_flat)
            next_values_flat = self.critic(next_states_flat)

        # 重塑为[T, N]格式
        values = values_flat.view(T, N)
        next_values = next_values_flat.view(T, N)

        # 确保所有张量在正确的设备上
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # 计算GAE
        advantages, returns = self.gae(rewards, dones, values, next_values)

        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Value Normalization
        returns_flat = returns.reshape(-1, 1)
        with torch.no_grad():
            self.value_norm.update(returns_flat)
        returns_normalized = self.value_norm.normalize(returns_flat).squeeze(-1)

        # 创建数据集
        dataset_size = T * N
        indices = torch.randperm(dataset_size, device=self.device)

        # 统计信息
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        clip_fraction = 0
        num_updates = 0

        # PPO更新
        with torch.enable_grad():
            for epoch in range(self.epochs):
                for start in range(0, dataset_size, self.batch_size):
                    end = min(start + self.batch_size, dataset_size)
                    batch_indices = indices[start:end]

                    # 获取批次数据
                    batch_states = states_flat[batch_indices]
                    batch_actions = actions_flat[batch_indices]
                    batch_advantages = advantages.reshape(-1)[batch_indices]
                    batch_returns = returns_normalized[batch_indices]

                    # 评估当前策略
                    new_log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)

                    # 获取旧策略的概率
                    with torch.no_grad():
                        old_log_probs, _ = self.actor.evaluate_action(batch_states, batch_actions)

                    # 计算比率
                    ratio = torch.exp(new_log_probs - old_log_probs)

                    # PPO Clip损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic损失
                    values_pred = self.critic(batch_states)
                    critic_loss = self.critic_loss_fn(values_pred, batch_returns)

                    # 熵损失
                    entropy_loss = -entropy.mean()

                    # 总损失
                    loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ 损失包含异常值: {loss.item()}")
                        continue  # 跳过这个批次

                    # 反向传播
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()

                    # 检查梯度
                    actor_has_nan = False
                    critic_has_nan = False
                    for param in self.actor.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            actor_has_nan = True
                            param.grad = torch.zeros_like(param.grad)

                    for param in self.critic.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            critic_has_nan = True
                            param.grad = torch.zeros_like(param.grad)

                    if actor_has_nan:
                        print("⚠️  Actor梯度包含NaN，已清零")
                    if critic_has_nan:
                        print("⚠️  Critic梯度包含NaN，已清零")

                    # 梯度裁剪 - 使用更保守的设置
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                    # 更新参数
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    # 统计信息
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy += entropy.mean().item()

                    # 计算KL散度
                    kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                    total_kl += kl

                    # 计算clip fraction
                    clip_fraction += ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                    num_updates += 1

        # 平均统计信息
        stats = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_divergence': total_kl / num_updates,
            'clip_fraction': clip_fraction / num_updates,
            'actor_grad_norm': actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm,
            'critic_grad_norm': critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm
        }

        self.update_count += 1
        return stats

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        选择动作

        Args:
            state: [state_dim] 状态
            deterministic: 是否使用确定性策略

        Returns:
            action: [action_dim] 动作
            log_prob: 对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state_tensor, deterministic)
            value = self.critic(state_tensor)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
            value.cpu().numpy()[0]
        )

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_norm_state_dict': self.value_norm.state_dict(),
            'update_count': self.update_count
        }, filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.value_norm.load_state_dict(checkpoint['value_norm_state_dict'])
        self.update_count = checkpoint['update_count']

        print(f"模型已从 {filepath} 加载")