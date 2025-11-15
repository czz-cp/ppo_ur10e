"""
UR10e PPO 多目标最优轨迹规划主训练程序

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》
实现了完整的PPO训练流程，包含：
1. 25维状态空间 + 6维动作空间
2. 多目标奖励函数（精度 + 平滑性 + 能耗）
3. 衰减回合机制（自适应课程学习）
4. PPO算法训练循环
5. 模型保存和可视化

运行方式：
python train.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 从utils导入工具类
try:
    from utils import (
        load_config, set_random_seed, validate_config,
        plot_training_curves, save_training_data,
        compute_trajectory_metrics, create_experiment_directory,
        save_experiment_config, compute_success_metrics,
        generate_training_report, TrainingLogger
    )
except ImportError:
    print("⚠️  无法从utils导入工具类，使用基本实现")

    def load_config(config_path="config.yaml"):
        """基本配置加载"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 确保数值类型正确转换
        if 'ppo' in config:
            ppo_config = config['ppo']
            # 转换学习率为浮点数
            if 'lr_actor' in ppo_config:
                ppo_config['lr_actor'] = float(ppo_config['lr_actor'])
            if 'lr_critic' in ppo_config:
                ppo_config['lr_critic'] = float(ppo_config['lr_critic'])
            # 转换其他数值参数
            for key in ['clip_eps', 'gamma', 'gae_lambda', 'entropy_coef', 'value_coef', 'max_grad_norm']:
                if key in ppo_config:
                    ppo_config[key] = float(ppo_config[key])
            for key in ['epochs', 'batch_size']:
                if key in ppo_config:
                    ppo_config[key] = int(ppo_config[key])

        return config

    def set_random_seed(seed=42):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def validate_config(config):
        """基本配置验证"""
        return True

    # 其他基本实现...
    plot_training_curves = lambda *args, **kwargs: None
    save_training_data = lambda *args, **kwargs: None
    compute_trajectory_metrics = lambda *args, **kwargs: {}
    create_experiment_directory = lambda *args, **kwargs: "./experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    save_experiment_config = lambda *args, **kwargs: None
    compute_success_metrics = lambda *args, **kwargs: {}
    generate_training_report = lambda *args, **kwargs: None

    class TrainingLogger:
        def __init__(self, log_dir="./logs"):
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)

        def log(self, message, level="INFO"):
            print(f"[{level}] {message}")

        def log_experiment_start(self, config):
            pass

        def log_experiment_end(self, final_stats):
            pass

from ur10e_env import UR10ePPOEnv
from ppo import PPO


def make_directories():
    """创建保存目录"""
    directories = [
        'checkpoints',      # 模型检查点
        'csv_output',       # CSV数据
        'plots',            # 训练曲线
        'trajectories',     # 轨迹可视化
        'logs'              # 日志文件
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ 目录结构创建完成")


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_trajectory_smoothness(trajectory_data: np.ndarray) -> float:
    """
    计算轨迹平滑度

    Args:
        trajectory_data: [T, 6] 关节角度序列

    Returns:
        smoothness: 平滑度指标（值越小越平滑）
    """
    if len(trajectory_data) < 3:
        return 0.0

    # 计算一阶差分（速度）
    velocity = np.diff(trajectory_data, axis=0)

    # 计算二阶差分（加速度）
    acceleration = np.diff(velocity, axis=0)

    # 平滑度指标：加速度的L2范数的平均值
    smoothness = np.mean(np.linalg.norm(acceleration, axis=1))

    return smoothness


def save_training_log(log_data: list, filepath: str):
    """保存训练日志"""
    with open(filepath, 'a', encoding='utf-8') as f:
        for entry in log_data:
            f.write(f"{entry}\n")


def main():
    """主训练函数"""
    print("=" * 80)
    print("🚀 UR10e PPO 多目标最优轨迹规划训练")
    print("基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》")
    print("=" * 80)

    # 创建目录
    make_directories()

    # 设置随机种子
    set_random_seed(42)

    # 加载配置文件
    try:
        config = load_config('config.yaml')
        print("✅ 成功加载 config.yaml")
    except Exception as e:
        print(f"⚠️  无法加载 config.yaml: {e}")
        print("使用默认配置...")
        config = {
            'env': {
                'xml_path': '../universal_robots_ur10e/ur10e_mujoco/scene.xml',
                'max_steps': 1000,
                'enable_rendering': False,
                'action_bound': 0.03,
                'dt': 0.01
            },
            'ppo': {
                'lr_actor': 3e-4,
                'lr_critic': 1e-3,
                'clip_eps': 0.2,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'epochs': 10,
                'batch_size': 64,
                'action_bound': 0.03
            },
            'train': {
                'max_episodes': 10000,
                'save_interval': 500,
                'log_interval': 100,
                'eval_interval': 1000
            },
            'decay_episode': {
                'enabled': True,
                'success_threshold': 0.7,
                'decay_window': 200
            }
        }

    # 保存配置
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"📝 配置已保存到 config.json")
    print(f"🔧 最大训练轮数: {config['train']['max_episodes']}")
    print(f"🎯 状态空间: 25维")
    print(f"⚡ 动作空间: 6维")
    print(f"📏 Action Bound: {config['ppo'].get('action_bound', config['env'].get('action_bound', 0.03))}")
    print(f"⏱️  时间步长 dt: {config['env'].get('dt', 0.01)}s")
    print(f"📊 衰减回合机制: {'启用' if config['decay_episode']['enabled'] else '禁用'}")
    print("-" * 80)

    # 创建环境
    print("🏗️  初始化环境...")
    env = UR10ePPOEnv(
        xml_path=config['env']['xml_path'],
        max_steps=config['env']['max_steps'],
        enable_rendering=config['env']['enable_rendering'],
        config=config  # 传递完整配置到环境
    )

    # 创建PPO智能体
    print("🧠 初始化PPO智能体...")
    agent = PPO(
        state_dim=25,
        action_dim=6,
        lr_actor=config['ppo']['lr_actor'],
        lr_critic=config['ppo']['lr_critic'],
        clip_eps=config['ppo']['clip_eps'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        entropy_coef=config['ppo']['entropy_coef'],
        value_coef=config['ppo']['value_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        epochs=config['ppo']['epochs'],
        batch_size=config['ppo']['batch_size'],
        action_bound=config['ppo'].get('action_bound', config['env'].get('action_bound', 0.03))
    )

    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rates': [],
        'position_errors': [],
        'decay_stats': [],
        'reward_components': []
    }

    # CSV数据存储
    csv_data = []

    print("🎯 开始训练...")
    print("-" * 80)

    # 训练循环
    for episode in range(1, config['train']['max_episodes'] + 1):
        # 重置环境
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0

        # 存储轨迹数据
        trajectory_data = [state[:6]]  # 存储关节角度
        states, actions, rewards, dones, next_states = [], [], [], [], []

        done = False
        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state, deterministic=False)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            trajectory_data.append(next_state[:6])

            episode_reward += reward
            episode_length += 1

            state = next_state

            # 每100步显示进度（仅在前几个episodes）
            if episode <= 3 and episode_length % 100 == 0:
                print(f"Episode {episode} | 步骤 {episode_length:4d} | "
                      f"误差: {info['pos_error']:.3f}m | "
                      f"奖励: {episode_reward:8.2f} | "
                      f"完成: {done}")

        # PPO更新
        if len(states) > 0:
            # 首先转换为numpy数组以提高性能
            states_array = np.array(states)
            actions_array = np.array(actions)
            rewards_array = np.array(rewards)
            dones_array = np.array(dones)
            next_states_array = np.array(next_states)

            # 创建张量并移动到正确的设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(device)  # [1, T, 25]
            actions_tensor = torch.FloatTensor(actions_array).unsqueeze(0).to(device)  # [1, T, 6]
            rewards_tensor = torch.FloatTensor(rewards_array).unsqueeze(0).to(device)  # [1, T]
            dones_tensor = torch.FloatTensor(dones_array).unsqueeze(0).to(device)  # [1, T]
            next_states_tensor = torch.FloatTensor(next_states_array).unsqueeze(0).to(device)  # [1, T, 25]

            # PPO更新
            update_stats = agent.update(
                states_tensor, actions_tensor, rewards_tensor,
                dones_tensor, next_states_tensor
            )

        # 衰减回合机制更新
        if config['decay_episode']['enabled']:
            env.update_decay_episode_mechanism(episode_reward, info['pos_error'])
            decay_stats = env.get_decay_stats()
        else:
            decay_stats = {}

        # 记录训练统计
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        training_stats['position_errors'].append(info['pos_error'])
        training_stats['decay_stats'].append(decay_stats)

        if 'reward_components' in info:
            training_stats['reward_components'].append(info['reward_components'])

        # 计算成功率（滑动窗口）
        window_size = min(100, len(training_stats['position_errors']))
        recent_errors = training_stats['position_errors'][-window_size:]
        success_count = sum(1 for error in recent_errors if error < 0.005)
        success_rate = success_count / len(recent_errors)
        training_stats['success_rates'].append(success_rate)

        # 保存CSV数据
        csv_row = {
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'pos_error': info['pos_error'],
            'success': 1 if info['pos_error'] < 0.005 else 0,
            'success_rate': success_rate,
            'decay_max_steps': decay_stats.get('current_max_steps', config['env']['max_steps'])
        }

        # 添加奖励分量
        if 'reward_components' in info:
            for key, value in info['reward_components'].items():
                csv_row[f'reward_{key}'] = value

        csv_data.append(csv_row)

        # 日志输出
        if episode % config['train']['log_interval'] == 0:
            # 计算平均指标
            avg_reward = np.mean(training_stats['episode_rewards'][-100:])
            avg_length = np.mean(training_stats['episode_lengths'][-100:])
            avg_error = np.mean(training_stats['position_errors'][-100:])

            print(f"Episode {episode:5d} | "
                  f"Reward: {episode_reward:8.2f} ({avg_reward:7.2f}) | "
                  f"Length: {episode_length:4d} ({avg_length:5.1f}) | "
                  f"Error: {info['pos_error']:.4f} ({avg_error:.4f}) | "
                  f"Success Rate: {success_rate:.2%} | "
                  f"Decay Steps: {decay_stats.get('current_max_steps', config['env']['max_steps'])}")

        # 保存模型
        if episode % config['train']['save_interval'] == 0:
            model_path = f"checkpoints/ppo_model_episode_{episode}.pt"
            agent.save(model_path)

        # 保存CSV数据
        if episode % 100 == 0:
            df = pd.DataFrame(csv_data)
            df.to_csv('csv_output/training_data.csv', index=False)

        # 绘制训练曲线
        if episode % config['train']['eval_interval'] == 0:
            plot_training_curves(training_stats)

    # 训练完成
    print("-" * 80)
    print("🎉 训练完成！")

    # 保存最终模型
    final_model_path = "checkpoints/ppo_model_final.pt"
    agent.save(final_model_path)

    # 保存最终CSV数据
    df = pd.DataFrame(csv_data)
    df.to_csv('csv_output/training_data_final.csv', index=False)

    # 绘制最终训练曲线
    plot_training_curves(training_stats, save_path='plots/final_training_curves.png')

    # 保存训练统计
    with open('logs/training_stats.json', 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型以便JSON序列化
        stats_for_json = {}
        for key, value in training_stats.items():
            if isinstance(value, list):
                stats_for_json[key] = [float(x) if isinstance(x, np.number) else x for x in value]
            else:
                stats_for_json[key] = value
        json.dump(stats_for_json, f, indent=2, ensure_ascii=False)

    print(f"💾 最终模型已保存: {final_model_path}")
    print(f"📊 训练数据已保存: csv_output/")
    print(f"📈 训练曲线已保存: plots/")
    print(f"📝 训练统计已保存: logs/")

    # 关闭环境
    env.close()

    print("=" * 80)
    print("✅ 训练流程全部完成！")
    print("=" * 80)


def plot_training_curves(training_stats: dict, save_path: str = None):
    """
    绘制训练曲线

    Args:
        training_stats: 训练统计数据
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Episode奖励
    axes[0, 0].plot(training_stats['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Episode长度
    axes[0, 1].plot(training_stats['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)

    # 位置误差
    axes[0, 2].plot(training_stats['position_errors'])
    axes[0, 2].set_title('Position Errors')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Error (m)')
    axes[0, 2].grid(True)
    axes[0, 2].axhline(y=0.005, color='r', linestyle='--', label='Success Threshold')
    axes[0, 2].legend()

    # 成功率
    axes[1, 0].plot(training_stats['success_rates'])
    axes[1, 0].set_title('Success Rate (100-episode window)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim([0, 1])

    # 平滑奖励
    window_size = min(100, len(training_stats['episode_rewards']))
    if window_size > 0:
        smooth_rewards = pd.Series(training_stats['episode_rewards']).rolling(window=window_size).mean()
        axes[1, 1].plot(smooth_rewards)
        axes[1, 1].set_title(f'Smoothed Rewards (window={window_size})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Smoothed Reward')
        axes[1, 1].grid(True)

    # 衰减回合统计
    if training_stats['decay_stats']:
        decay_steps = [stats.get('current_max_steps', 1000) for stats in training_stats['decay_stats']]
        axes[1, 2].plot(decay_steps)
        axes[1, 2].set_title('Decaying Episode Max Steps')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Max Steps')
        axes[1, 2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线已保存: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()