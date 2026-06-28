import json
from sys import argv
import argparse

# 导入Manager类，用于管理训练流程
from manager import Manager
# 导入ArchitectureConfig类，用于解析和存储模型结构配置
from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig


def read_json_file(path):
    # 打开并读取文件内容，去除回车和换行符
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        # 尝试将字符串解析为JSON对象
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_training_copy.json', help='配置文件路径')
    args = parser.parse_args()
    path = args.config

    config = read_json_file(path)
    # 使用ArchitectureConfig解析配置，生成训练参数对象
    training_parameters = ArchitectureConfig.parse_obj(config)
    # 创建Manager对象，传入训练参数
    manager = Manager(training_parameters)
    # 执行训练流程
    manager.execute()
