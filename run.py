#!/usr/bin/env python3
"""
UR10e PPO 多目标最优轨迹规划启动脚本

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》
简化版启动脚本，包含基本的错误处理和配置验证

使用方法:
    python run.py
"""

import os
import sys
import traceback

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from utils import load_config, set_random_seed, validate_config, TrainingLogger
    from train import main as train_main

    def main():
        """主函数"""
        print("=" * 80)
        print("🚀 UR10e PPO 多目标最优轨迹规划")
        print("基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》")
        print("=" * 80)

        # 加载配置
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return

        print(f"📋 加载配置文件: {config_path}")
        config = load_config(config_path)

        # 验证配置
        if not validate_config(config):
            print("❌ 配置文件验证失败")
            return

        # 设置随机种子
        seed = config.get('seed', 42)
        set_random_seed(seed)

        # 创建日志记录器
        logger = TrainingLogger()
        logger.log_experiment_start(config)

        try:
            # 开始训练
            print("🎯 开始训练...")
            train_main()

            logger.log("🎉 训练完成！")
            logger.log("📁 结果已保存到相应目录")

        except KeyboardInterrupt:
            print("\n⚠️  训练被用户中断")
            logger.log("训练被用户中断", "WARNING")

        except Exception as e:
            print(f"❌ 训练过程中发生错误: {e}")
            logger.log(f"训练错误: {str(e)}", "ERROR")
            logger.log(f"错误详情: {traceback.format_exc()}", "ERROR")

        finally:
            print("=" * 80)
            print("✅ 程序执行完成")
            print("=" * 80)

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有依赖文件都在正确位置:")
    print("- ur10e_env.py")
    print("- ppo.py")
    print("- utils.py")
    print("- train.py")
    print("- config.yaml")
    sys.exit(1)

except Exception as e:
    print(f"❌ 启动失败: {e}")
    print(f"错误详情: {traceback.format_exc()}")
    sys.exit(1)