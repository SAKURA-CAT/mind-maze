"""
@DATE: 2025-05-27 19:01:38
@File: eval.py
@IDE: vscode
@Description:
    基于 evalscope 评估常规 Qwen3 模型的输出精度
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用的GPU设备
from evalscope.run import run_task
from evalscope import TaskConfig

task_cfg = TaskConfig(
    model="./weights/Qwen3-4B",
    datasets=[
        "data_collection",
    ],
    dataset_args={
        "data_collection": {
            "dataset_id": "modelscope/EvalScope-Qwen3-Test",
            "filters": {"remove_until": "</think>"},  # 过滤掉思考的内容
        }
    },
    eval_batch_size=48,
    generation_config={
        "max_tokens": 2048,  # 最大生成token数
        "temperature": 0.6,  # 采样温度 (qwen 报告推荐值)
        "top_p": 0.95,  # top-p采样 (qwen 报告推荐值)
        "top_k": 20,  # top-k采样 (qwen 报告推荐值)
        "n": 1,  # 每个请求产生的回复数量
        "chat_template_kwargs": {"enable_thinking": False},  # 关闭思考模式
    },
    limit=3000,
)

run_task(task_cfg=task_cfg)
