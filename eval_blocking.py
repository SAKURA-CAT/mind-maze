"""
@DATE: 2025-05-28 13:25:21
@File: eval/blocking.py
@IDE: vscode
@Description:
    测试 Qwen3ForCausalLM 的分块混淆思考模式
"""

from evalscope.run import run_task
from evalscope import TaskConfig
from model_blocking import MODE_NAME

task_cfg = TaskConfig(
    model=MODE_NAME,
    datasets=[
        "data_collection",
    ],
    dataset_args={
        "data_collection": {
            "dataset_id": "modelscope/EvalScope-Qwen3-Test",
            "filters": {"remove_until": "</think>"},  # 过滤掉思考的内容
        }
    },
    # 魔改后的Qwen3模型只能处理一个 batch
    eval_batch_size=1,
    generation_config={
        "max_tokens": 10000,  # 最大生成token数
        "temperature": 0.6,  # 采样温度 (qwen 报告推荐值)
        "top_p": 0.95,  # top-p采样 (qwen 报告推荐值)
        "top_k": 20,  # top-k采样 (qwen 报告推荐值)
        "n": 1,  # 每个请求产生的回复数量
        "chat_template_kwargs": {"enable_thinking": False},  # 关闭思考模式
    },
    limit=100,
)

run_task(task_cfg=task_cfg)
