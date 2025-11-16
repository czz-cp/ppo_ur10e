#!/usr/bin/env python3

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ur10e_env import UR10ePPOEnv as UR10eEnvironment
from utils import RewardNormalizer
import matplotlib.pyplot as plt

def test_reward_normalizer():
    """测试奖励归一化器功能"""
    print("🧪 测试奖励归一化器...")

    # 创建归一化器
    normalizer = RewardNormalizer(
        gamma=0.99,
        clip_range=5.0,
        normalize_method='running_stats',
        warmup_steps=10
    )

    # 模拟奖励序列
    np.random.seed(42)
    raw_rewards = np.random.randn(100) + 2.0  # 均值为2的奖励
    raw_rewards[:20] = np.random.randn(20) - 1.0  # 前20个奖励均值为-1
    raw_rewards[40:60] = np.random.randn(20) + 5.0  # 中间20个奖励均值为5

    normalized_rewards = []

    print(f"📊 奖励归一化统计:")
    print(f"原始奖励统计: 均值={np.mean(raw_rewards):.3f}, 标准差={np.std(raw_rewards):.3f}")
    print(f"原始奖励范围: [{np.min(raw_rewards):.3f}, {np.max(raw_rewards):.3f}]")

    for i, reward in enumerate(raw_rewards):
        # 更新归一化器
        normalizer.update(reward, done=(i % 25 == 24))  # 每25步算一个episode结束

        # 归一化奖励
        normalized_reward = normalizer.normalize(reward)
        normalized_rewards.append(normalized_reward)

        if i < 10:
            print(f"步骤 {i:2d}: 原始={reward:6.3f} -> 归一化={normalized_reward:6.3f}")

    normalized_rewards = np.array(normalized_rewards)

    print(f"\n📈 归一化后奖励统计:")
    print(f"归一化奖励统计: 均值={np.mean(normalized_rewards):.3f}, 标准差={np.std(normalized_rewards):.3f}")
    print(f"归一化奖励范围: [{np.min(normalized_rewards):.3f}, {np.max(normalized_rewards):.3f}]")

    # 获取归一化器统计
    stats = normalizer.get_stats()
    print(f"\n🔧 归一化器最终统计:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 测试不同归一化方法
    print(f"\n🎯 测试不同归一化方法:")
    methods = ['running_stats', 'batch_stats', 'rank']

    for method in methods:
        normalizer_method = RewardNormalizer(
            normalize_method=method,
            warmup_steps=10
        )

        test_rewards = [1.0, 2.0, 3.0, -1.0, 5.0]

        print(f"\n  方法: {method}")
        for i, reward in enumerate(test_rewards):
            normalizer_method.update(reward)
            normalized = normalizer_method.normalize(reward)
            print(f"    {reward:5.1f} -> {normalized:6.3f}")

    return True

def test_environment_reward_normalization():
    """测试环境中的奖励归一化"""
    print(f"\n🤖 测试环境奖励归一化...")

    try:
        # 创建环境
        env = UR10eEnvironment()

        # 运行几个测试步骤
        print(f"🎬 运行测试回合...")

        raw_rewards = []
        normalized_rewards = []

        for episode in range(3):
            state = env.reset()
            episode_raw = []
            episode_normalized = []

            print(f"\n回合 {episode + 1}:")

            for step in range(50):
                # 随机动作
                action = env.action_space.sample()

                # 执行动作
                next_state, reward, done, info = env.step(action)

                # 收集奖励信息
                if 'raw_reward' in info['reward_components']:
                    raw_reward = info['reward_components']['raw_reward']
                    normalized_reward = info['reward_components']['normalized_reward']

                    episode_raw.append(raw_reward)
                    episode_normalized.append(normalized_reward)

                    if step < 5:  # 只显示前5步
                        print(f"  步骤 {step:2d}: 原始={raw_reward:8.4f} -> 归一化={normalized_reward:8.4f}")

                if done:
                    break

            if episode_raw:
                print(f"  回合统计: 原始奖励均值={np.mean(episode_raw):.4f}, 归一化奖励均值={np.mean(episode_normalized):.4f}")
                raw_rewards.extend(episode_raw)
                normalized_rewards.extend(episode_normalized)

        # 显示总体统计
        if raw_rewards:
            print(f"\n📊 总体奖励统计:")
            print(f"原始奖励: 均值={np.mean(raw_rewards):.4f}, 标准差={np.std(raw_rewards):.4f}, 范围=[{np.min(raw_rewards):.4f}, {np.max(raw_rewards):.4f}]")
            print(f"归一化奖励: 均值={np.mean(normalized_rewards):.4f}, 标准差={np.std(normalized_rewards):.4f}, 范围=[{np.min(normalized_rewards):.4f}, {np.max(normalized_rewards):.4f}]")

        # 显示归一化器统计
        stats = env.get_reward_normalizer_stats()
        print(f"\n🔧 环境归一化器统计:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # 测试切换归一化方法
        print(f"\n🔄 测试切换归一化方法...")
        env.set_reward_normalization_method('batch_stats')
        env.set_reward_normalization_method('rank')
        env.set_reward_normalization_method('running_stats')  # 切换回默认方法

        return True

    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_reward_normalization_comparison():
    """绘制奖励归一化对比图"""
    print(f"\n📈 生成奖励归一化对比图...")

    try:
        # 生成测试数据
        np.random.seed(42)
        raw_rewards = np.random.randn(200) * 2 + 1.0  # 均值1，标准差2

        # 添加一些异常值
        raw_rewards[50:60] = np.random.randn(10) * 5 + 10.0  # 高奖励区间
        raw_rewards[100:110] = np.random.randn(10) * 3 - 8.0  # 低奖励区间

        methods = ['running_stats', 'batch_stats', 'rank']
        normalized_results = {}

        for method in methods:
            normalizer = RewardNormalizer(
                normalize_method=method,
                warmup_steps=20
            )

            normalized = []
            for reward in raw_rewards:
                normalizer.update(reward)
                normalized.append(normalizer.normalize(reward))

            normalized_results[method] = np.array(normalized)

        # 绘制对比图
        plt.figure(figsize=(15, 10))

        # 原始奖励
        plt.subplot(2, 2, 1)
        plt.plot(raw_rewards, 'b-', alpha=0.7, label='Raw Rewards')
        plt.title('Raw Rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()

        # 不同归一化方法对比
        colors = ['r-', 'g-', 'm-']
        for i, method in enumerate(methods):
            plt.subplot(2, 2, i + 2)
            plt.plot(normalized_results[method], colors[i], alpha=0.7, label=f'{method}')
            plt.title(f'Normalized Rewards ({method})')
            plt.xlabel('Step')
            plt.ylabel('Normalized Reward')
            plt.grid(True)
            plt.legend()
            plt.ylim([-6, 6])

        plt.tight_layout()

        # 保存图像
        save_path = "reward_normalization_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 对比图已保存到: {save_path}")

        return True

    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始奖励归一化测试...")

    success1 = test_reward_normalizer()
    success2 = test_environment_reward_normalization()
    success3 = plot_reward_normalization_comparison()

    if success1 and success2 and success3:
        print("\n🎉 所有奖励归一化测试通过！")
        print("✅ 奖励归一化功能已成功集成到PPO系统中！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1)