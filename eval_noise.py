"""
@DATE: 2025-05-29 23:32:15
@File: eval/noise.py
@IDE: vscode
@Description:
    测试随机噪声对模型的影响
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置使用的GPU设备

from evalscope.run import run_task
from evalscope import TaskConfig
from model_noise import MODEL_NAME

task_cfg = TaskConfig(
    model=MODEL_NAME,
    datasets=[
        "data_collection",
    ],
    dataset_args={
        "data_collection": {
            "dataset_id": "modelscope/EvalScope-Qwen3-Test",
            "filters": {"remove_until": "</think>"},  # 过滤掉思考的内容
        }
    },
    eval_batch_size=32,
    generation_config={
        "max_tokens": 3276,  # 最大生成token数
        "temperature": 0.6,  # 采样温度 (qwen 报告推荐值)
        "top_p": 0.95,  # top-p采样 (qwen 报告推荐值)
        "top_k": 20,  # top-k采样 (qwen 报告推荐值)
        "use_cache": True,
        "n": 1,  # 每个请求产生的回复数量
        "chat_template_kwargs": {"enable_thinking": True},
    },
    limit=1000,
)

run_task(task_cfg=task_cfg)
