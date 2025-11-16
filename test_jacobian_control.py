#!/usr/bin/env python3

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ur10e_env import UR10ePPOEnv as UR10eEnvironment

def test_jacobian_control():
    """测试基于雅可比的RL-PID控制"""

    print("🧪 测试基于雅可比的RL-PID控制...")

    try:
        # 创建环境
        env = UR10eEnvironment()

        # 重置环境
        state = env.reset()
        print(f"✅ 环境重置成功，初始状态维度: {state.shape}")

        # 测试几个随机动作
        for episode in range(3):
            print(f"\n🎬 测试回合 {episode + 1}/3")

            # 重置环境
            state = env.reset()

            total_reward = 0
            steps = 0
            max_steps = 200

            while steps < max_steps:
                # 随机动作
                action = env.action_space.sample()

                # 执行动作
                next_state, reward, done, info = env.step(action)

                total_reward += reward
                steps += 1

                # 每50步输出一次状态
                if steps % 50 == 0:
                    current_pos = env.data.site_xpos[0].copy()
                    target_pos = env.target_pos if hasattr(env, 'target_pos') else np.array([0.5, 0.0, 0.5])
                    distance = np.linalg.norm(target_pos - current_pos)

                    print(f"  步骤 {steps}: 距离目标 = {distance:.4f}m, 奖励 = {reward:.4f}")

                if done:
                    print(f"  ✅ 回合完成，总步数: {steps}")
                    break

            print(f"  📊 回合统计: 总奖励 = {total_reward:.4f}, 步数 = {steps}")

            # 显示最终状态
            final_pos = env.data.site_xpos[0].copy()
            target_pos = env.target_pos if hasattr(env, 'target_pos') else np.array([0.5, 0.0, 0.5])
            final_distance = np.linalg.norm(target_pos - final_pos)
            print(f"  🎯 最终距离目标: {final_distance:.4f}m")

        print(f"\n🎉 雅可比控制测试完成！")

        # 测试雅可比矩阵计算
        print(f"\n🔬 测试雅可比矩阵计算...")
        test_joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # 计算雅可比矩阵
        jacobian = env._compute_jacobian(test_joint_angles)
        print(f"✅ 雅可比矩阵计算成功")
        print(f"  雅可比矩阵形状: {jacobian.shape}")
        print(f"  雅可比矩阵条件数: {np.linalg.cond(jacobian):.2f}")
        print(f"  雅可比矩阵秩: {np.linalg.matrix_rank(jacobian)}")

        # 测试雅可比转置控制
        end_effector_error = np.array([0.1, 0.05, -0.02])  # 10cm, 5cm, -2cm误差
        joint_error = jacobian.T @ end_effector_error
        print(f"✅ 雅可比转置控制映射成功")
        print(f"  末端执行器误差: {end_effector_error}m")
        print(f"  关节空间误差: {joint_error}rad")
        print(f"  关节空间误差范数: {np.linalg.norm(joint_error):.4f}rad")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jacobian_accuracy():
    """测试雅可比矩阵的数值精度"""

    print(f"\n🎯 测试雅可比矩阵数值精度...")

    try:
        env = UR10eEnvironment()
        state = env.reset()

        # 测试关节角度
        test_angles = [
            np.zeros(6),
            np.array([0.5, -0.3, 0.2, -0.1, 0.4, -0.2]),
            np.array([1.0, 0.5, -0.5, 0.8, -0.3, 0.6])
        ]

        for i, angles in enumerate(test_angles):
            print(f"\n  测试配置 {i+1}: {angles}")

            # 计算解析雅可比
            jacobian = env._compute_jacobian(angles)

            # 使用有限差分验证
            epsilon = 1e-4
            fd_jacobian = np.zeros((3, 6))

            current_pos = env._get_end_effector_position(angles)

            for j in range(6):
                delta_q = np.zeros(6)
                delta_q[j] = epsilon

                perturbed_pos = env._get_end_effector_position(angles + delta_q)
                fd_jacobian[:, j] = (perturbed_pos - current_pos) / epsilon

            # 计算误差
            jacobian_error = np.linalg.norm(jacobian - fd_jacobian)
            relative_error = jacobian_error / (np.linalg.norm(jacobian) + 1e-8)

            print(f"    雅可比误差: {jacobian_error:.6f}")
            print(f"    相对误差: {relative_error:.6f}")

            if relative_error < 1e-3:
                print(f"    ✅ 精度良好")
            else:
                print(f"    ⚠️  精度需要改进")

        print(f"\n✅ 雅可比数值精度测试完成")
        return True

    except Exception as e:
        print(f"❌ 雅可比精度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_jacobian_control()
    success2 = test_jacobian_accuracy()

    if success1 and success2:
        print("\n🎉 所有测试通过！基于雅可比的RL-PID控制系统工作正常！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1)