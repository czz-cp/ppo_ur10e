"""
UR10e Forward and Inverse Kinematics

基于改进的FK-IK-ur10e代码，提供精确的UR10e运动学解算
包含完整的正向运动学(FK)和逆向运动学(IK)实现

单位：米
"""

import numpy as np
import math
from typing import List, Tuple, Optional


class UR10eKinematics:
    """
    UR10e运动学解算器

    基于标准D-H参数实现完整的正向和逆向运动学
    """

    def __init__(self):
        # UR10e D-H参数 (单位：米)
        # 基于官方UR10e技术规格
        self.a = [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0]
        self.d = [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655]
        self.alpha = [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0]

        # 关节限制 (弧度)
        self.joint_limits = [
            [-3.14159, 3.14159],   # shoulder_pan_joint
            [-3.14159, 3.14159],   # shoulder_lift_joint
            [-3.14159, 3.14159],   # elbow_joint
            [-3.14159, 3.14159],   # wrist_1_joint
            [-3.14159, 3.14159],   # wrist_2_joint
            [-3.14159, 3.14159]    # wrist_3_joint
        ]

    def _transformation_matrix(self, theta: float, a: float, d: float, alpha: float) -> np.ndarray:
        """
        计算D-H变换矩阵

        Args:
            theta: 关节角度 (弧度)
            a: 连杆长度 (米)
            d: 连杆偏移 (米)
            alpha: 扭转角 (弧度)

        Returns:
            T: 4x4变换矩阵
        """
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        T = np.array([
            [ct,    -st*ca,  st*sa, a*ct],
            [st,     ct*ca, -ct*sa, a*st],
            [0.0,      sa,     ca,   d  ],
            [0.0,     0.0,    0.0,  1.0]
        ])

        return T

    def forward_kinematics(self, joint_angles: List[float]) -> np.ndarray:
        """
        正向运动学计算

        Args:
            joint_angles: 6个关节角度 [弧度]

        Returns:
            T06: 末端执行器相对于基座的变换矩阵 (4x4)
        """
        if len(joint_angles) != 6:
            raise ValueError("需要6个关节角度")

        # 验证关节限制
        for i, angle in enumerate(joint_angles):
            if not (self.joint_limits[i][0] <= angle <= self.joint_limits[i][1]):
                print(f"警告: 关节{i+1}角度{angle:.3f}超出限制范围")

        # 计算各关节变换矩阵
        T01 = self._transformation_matrix(joint_angles[0], self.a[0], self.d[0], self.alpha[0])
        T12 = self._transformation_matrix(joint_angles[1], self.a[1], self.d[1], self.alpha[1])
        T23 = self._transformation_matrix(joint_angles[2], self.a[2], self.d[2], self.alpha[2])
        T34 = self._transformation_matrix(joint_angles[3], self.a[3], self.d[3], self.alpha[3])
        T45 = self._transformation_matrix(joint_angles[4], self.a[4], self.d[4], self.alpha[4])
        T56 = self._transformation_matrix(joint_angles[5], self.a[5], self.d[5], self.alpha[5])

        # 计算总变换矩阵
        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

        return T06

    def get_end_effector_position(self, joint_angles: List[float]) -> np.ndarray:
        """
        获取末端执行器位置

        Args:
            joint_angles: 6个关节角度 [弧度]

        Returns:
            position: [x, y, z] 位置 [米]
        """
        T06 = self.forward_kinematics(joint_angles)
        position = T06[:3, 3]
        return position

    def get_end_effector_orientation(self, joint_angles: List[float]) -> np.ndarray:
        """
        获取末端执行器姿态

        Args:
            joint_angles: 6个关节角度 [弧度]

        Returns:
            orientation: 3x3 旋转矩阵
        """
        T06 = self.forward_kinematics(joint_angles)
        orientation = T06[:3, :3]
        return orientation

    def inverse_kinematics(self, target_pose: np.ndarray) -> List[List[float]]:
        """
        逆向运动学计算 (解析解)

        Args:
            target_pose: 目标位姿 4x4 变换矩阵

        Returns:
            solutions: 所有可能解的列表 [[theta1, theta2, ..., theta6], ...]
        """
        if target_pose.shape != (4, 4):
            raise ValueError("目标位姿必须是4x4矩阵")

        # 提取目标位姿元素
        nx, ny, nz = target_pose[0, 0], target_pose[1, 0], target_pose[2, 0]
        ox, oy, oz = target_pose[0, 1], target_pose[1, 1], target_pose[2, 1]
        ax, ay, az = target_pose[0, 2], target_pose[1, 2], target_pose[2, 2]
        px, py, pz = target_pose[0, 3], target_pose[1, 3], target_pose[2, 3]

        solutions = []

        # 求解theta1 (shoulder_pan_joint)
        m = self.d[5] * ay - py
        n = ax * self.d[5] - px

        denominator = m**2 + n**2 - self.d[3]**2
        if denominator < 0:
            return solutions  # 无解

        sqrt_val = math.sqrt(denominator)

        theta1_candidates = [
            math.atan2(m, n) - math.atan2(self.d[3], sqrt_val),
            math.atan2(m, n) - math.atan2(self.d[3], -sqrt_val)
        ]

        for theta1 in theta1_candidates:
            # 求解theta5 (wrist_3_joint)
            sin_theta5 = ax * math.sin(theta1) - ay * math.cos(theta1)

            # 检查是否在有效范围内
            if abs(sin_theta5) > 1.0:
                continue

            theta5_candidates = [
                math.acos(sin_theta5),
                -math.acos(sin_theta5)
            ]

            for theta5 in theta5_candidates:
                # 求解theta6 (wrist_2_joint)
                mm = nx * math.sin(theta1) - ny * math.cos(theta1)
                nn = ox * math.sin(theta1) - oy * math.cos(theta1)

                # 检查分母不为零
                if abs(math.cos(theta5)) < 1e-6:
                    theta6 = 0.0  # 奇异位置
                else:
                    theta6 = math.atan2(mm, nn) - math.atan2(math.sin(theta5), 0)

                # 求解theta3 (elbow_joint)
                m_val = (self.d[4] * (math.sin(theta6) * (nx*math.cos(theta1) + ny*math.sin(theta1)) +
                           math.cos(theta6) * (ox*math.cos(theta1) + oy*math.sin(theta1))) -
                        self.d[5] * (ax*math.cos(theta1) + ay*math.sin(theta1)) +
                        px*math.cos(theta1) + py*math.sin(theta1))

                n_val = (pz - self.d[0] - az*self.d[5] +
                        self.d[4] * (oz*math.cos(theta6) + nz*math.sin(theta6)))

                # 检查elbow关节是否有解
                elbow_val = (m_val**2 + n_val**2 - self.a[1]**2 - self.a[2]**2) / (2 * self.a[1] * self.a[2])

                if abs(elbow_val) > 1.0:
                    continue

                theta3_candidates = [
                    math.acos(elbow_val),
                    -math.acos(elbow_val)
                ]

                for theta3 in theta3_candidates:
                    # 求解theta2 (shoulder_lift_joint)
                    s2 = ((self.a[2] * math.cos(theta3) + self.a[1]) * n_val -
                          self.a[2] * math.sin(theta3) * m_val) / \
                         (self.a[1]**2 + self.a[2]**2 + 2 * self.a[1] * self.a[2] * math.cos(theta3))

                    c2 = (m_val + self.a[2] * math.sin(theta3) * s2) / \
                         (self.a[2] * math.cos(theta3) + self.a[1])

                    theta2 = math.atan2(s2, c2)

                    # 求解theta4 (wrist_1_joint)
                    numerator = (-math.sin(theta6) * (nx*math.cos(theta1) + ny*math.sin(theta1)) -
                                math.cos(theta6) * (ox*math.cos(theta1) + oy*math.sin(theta1)))
                    denominator = oz*math.cos(theta6) + nz*math.sin(theta6)

                    theta4 = math.atan2(numerator, denominator) - theta2 - theta3

                    # 验证解的有效性
                    solution = [theta1, theta2, theta3, theta4, theta5, theta6]

                    # 检查关节限制
                    valid = True
                    for i, angle in enumerate(solution):
                        if not (self.joint_limits[i][0] - 0.1 <= angle <= self.joint_limits[i][1] + 0.1):
                            valid = False
                            break

                    if valid:
                        solutions.append(solution)

        return solutions

    def inverse_kinematics_position(self, target_position: np.ndarray) -> List[List[float]]:
        """
        仅基于位置的逆运动学 (简化版本)

        Args:
            target_position: 目标位置 [x, y, z] (米)

        Returns:
            solutions: 所有可能解的列表
        """
        # 使用默认姿态 (末端垂直向下)
        default_orientation = np.array([
            [0, 0, 1, target_position[0]],
            [0, 1, 0, target_position[1]],
            [-1, 0, 0, target_position[2]],
            [0, 0, 0, 1]
        ])

        return self.inverse_kinematics(default_orientation)

    def select_best_solution(self, solutions: List[List[float]],
                           current_angles: List[float],
                           preferred_solution: int = 0) -> Optional[List[float]]:
        """
        从多个解中选择最优解

        Args:
            solutions: IK解的列表
            current_angles: 当前关节角度
            preferred_solution: 偏好的解索引 (0-7)

        Returns:
            best_solution: 最优解
        """
        if not solutions:
            return None

        # 如果有偏好解且有效，直接返回
        if preferred_solution < len(solutions):
            return solutions[preferred_solution]

        # 否则选择与当前角度最接近的解
        best_idx = 0
        min_distance = float('inf')

        for i, solution in enumerate(solutions):
            distance = sum(abs(solution[j] - current_angles[j]) for j in range(6))
            if distance < min_distance:
                min_distance = distance
                best_idx = i

        return solutions[best_idx]


def test_kinematics():
    """测试运动学解算器"""
    kinematics = UR10eKinematics()

    # 测试正向运动学
    test_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print("测试关节角度:", test_angles)

    T = kinematics.forward_kinematics(test_angles)
    position = kinematics.get_end_effector_position(test_angles)
    print("正向运动学位置:", position)
    print("变换矩阵:\n", T)

    # 测试逆向运动学
    print("\n测试逆向运动学:")
    solutions = kinematics.inverse_kinematics(T)
    print(f"找到 {len(solutions)} 个解:")

    for i, solution in enumerate(solutions):
        print(f"解 {i+1}: {[angle*180/math.pi for angle in solution]} (度)")

        # 验证解的正确性
        computed_T = kinematics.forward_kinematics(solution)
        position_error = np.linalg.norm(computed_T[:3, 3] - position)
        print(f"  位置误差: {position_error*1000:.2f} mm")


if __name__ == "__main__":
    test_kinematics()