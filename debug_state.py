#!/usr/bin/env python3

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_state_calculation():
    """测试状态计算的各个组件"""

    print("🧪 测试状态计算组件...")

    # 模拟输入数据
    current_joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    current_joint_velocities = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    current_end_pos = np.array([0.5, 0.3, 0.4])
    target_pos = np.array([0.6, 0.2, 0.3])

    print(f"✅ 输入数据初始化成功")
    print(f"  current_joint_angles: {current_joint_angles.shape} {current_joint_angles}")
    print(f"  current_joint_velocities: {current_joint_velocities.shape} {current_joint_velocities}")
    print(f"  current_end_pos: {current_end_pos.shape} {current_end_pos}")
    print(f"  target_pos: {target_pos.shape} {target_pos}")

    # 计算位置误差
    position_error = target_pos - current_end_pos
    print(f"✅ position_error: {position_error.shape} {position_error}")

    # 计算欧氏距离
    distance_to_target = np.linalg.norm(target_pos - current_end_pos)
    print(f"✅ distance_to_target: {distance_to_target} (type: {type(distance_to_target)}, shape: {np.array(distance_to_target).shape})")

    # 测试归一化函数（简化���本）
    def normalize_joint_positions(positions):
        max_positions = np.array([6.283, 6.283, 3.142, 6.283, 6.283, 6.283])
        return np.clip(positions / max_positions, -1.0, 1.0)

    def normalize_joint_velocities(velocities):
        max_velocities = np.array([2.094, 2.094, 3.142, 3.142, 3.142, 3.142])
        return np.clip(velocities / max_velocities, -1.0, 1.0)

    def normalize_position_error(error):
        return np.clip(error / 1.0, -1.0, 1.0)

    def normalize_distance(distance):
        return np.clip(distance / 2.0, 0.0, 1.0)

    # 测试归一化
    try:
        norm_angles = normalize_joint_positions(current_joint_angles)
        print(f"✅ normalized angles: {norm_angles.shape} {norm_angles}")

        norm_velocities = normalize_joint_velocities(current_joint_velocities)
        print(f"✅ normalized velocities: {norm_velocities.shape} {norm_velocities}")

        norm_error = normalize_position_error(position_error)
        print(f"✅ normalized error: {norm_error.shape} {norm_error}")

        norm_distance = normalize_distance(distance_to_target)
        print(f"✅ normalized distance: {norm_distance} (type: {type(norm_distance)})")

        # 测试concatenate
        normalized_state = np.concatenate([
            norm_angles,            # 6维
            norm_velocities,        # 6维
            norm_error,            # 3维
            [norm_distance]         # 1维，包装为数组
        ])

        print(f"✅ 最终状态向量: {normalized_state.shape} {normalized_state}")
        print(f"✅ 状态向量范围: [{normalized_state.min():.3f}, {normalized_state.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ 归一化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_state_calculation()
    if success:
        print("\n🎉 状态计算测试通过！")
    else:
        print("\n❌ 状态计算测试失败！")
        sys.exit(1)