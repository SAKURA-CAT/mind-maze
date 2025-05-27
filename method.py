"""
@DATE: 2025-05-27 18:28:24
@File: method.py
@IDE: vscode
@Description:
    定义混淆算法
"""

import torch


def adaptive_rearrange(tensor: torch.Tensor, config: dict) -> torch.Tensor:
    """
    根据强度参数动态重组二维张量

    参数：
    tensor: 形状为 [1, seq_len] 的二维张量
    intensity: 重组强度 (0=不重组, 1=完全打乱)

    返回：
    重组后的张量（保持原始形状）
    """
    intensity = config.get("intensity", 0.5)
    assert 0 <= intensity <= 1, "强度参数必须在0-1之间"
    assert tensor.dim() == 2 and tensor.size(0) == 1, "输入必须是形状[1, seq_len]"

    seq_len = tensor.size(1)
    if seq_len <= 1 or intensity == 0:
        return tensor.clone()

    # 计算切分块数（强度越大，块数越多）
    min_chunks = 1
    max_chunks = seq_len
    num_chunks = max(min_chunks, min(max_chunks, int(1 / intensity)))

    # 计算每个块的平均长度
    chunk_size = max(1, seq_len // num_chunks)

    # 切分张量
    chunks = []
    for i in range(0, seq_len, chunk_size):
        chunk = tensor[:, i : i + chunk_size]
        if chunk.size(1) > 0:  # 跳过空块
            chunks.append(chunk)

    # 随机重排块顺序
    if len(chunks) > 1:
        perm = torch.randperm(len(chunks))
        chunks = [chunks[i] for i in perm]

    # 合并结果
    return torch.cat(chunks, dim=1)


methods = {
    "adaptive_rearrange": adaptive_rearrange,
}


if __name__ == "__main__":
    # 测试代码
    test_tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    intensity = 0.5
    result = adaptive_rearrange(test_tensor, intensity)
    print("原始张量:", test_tensor)
    print("重组后的张量:", result)
