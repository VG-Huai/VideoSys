import torch
import torch.nn.functional as F
import xformers.ops as xops

# 设置设备
device = "cuda"

# 生成随机输入数据
batch_size, num_heads, seq_len, dim_head = 2, 4, 16, 640

# 选择数据类型
dtypes = [torch.float16, torch.float32, torch.bfloat16]

for dtype in dtypes:
    print(f"\nTesting with dtype: {dtype}\n" + "="*50)

    # 生成随机的 q, k, v
    q = torch.randn(batch_size, num_heads, seq_len, dim_head, device=device, dtype=dtype) * 5
    k = torch.randn(batch_size, num_heads, seq_len, dim_head, device=device, dtype=dtype) * 5
    v = torch.randn(batch_size, num_heads, seq_len, dim_head, device=device, dtype=dtype) * 5

    scale = 1.0 / (dim_head ** 0.5)

    # ✅ 方法 1: PyTorch 标准实现（完整 attention 计算）
    attn_pytorch = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    # ✅ 方法 2: Einsum 实现（完整 attention 计算）
    attn_einsum_weights = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    attn_einsum_weights = attn_einsum_weights.softmax(dim=-1)  # 计算 softmax(QK^T)
    attn_einsum = torch.einsum("b h i j, b h j d -> b h i d", attn_einsum_weights, v)  # 计算 softmax(QK^T) @ V

    # ✅ 方法 3: Matmul 实现（完整 attention 计算）
    attn_matmul_weights = (q @ k.transpose(-2, -1)) * scale
    attn_matmul_weights = attn_matmul_weights.softmax(dim=-1)  # 计算 softmax(QK^T)
    attn_matmul = attn_matmul_weights @ v  # 计算 softmax(QK^T) @ V

    # ✅ 方法 4: xFormers 实现（完整 attention 计算）
    q_flat = q.reshape(batch_size * num_heads, seq_len, dim_head)
    k_flat = k.reshape(batch_size * num_heads, seq_len, dim_head)
    v_flat = v.reshape(batch_size * num_heads, seq_len, dim_head)

    attn_xformers = xops.memory_efficient_attention(q_flat, k_flat, v_flat, attn_bias=None, op=None)
    attn_xformers = attn_xformers.reshape(batch_size, num_heads, seq_len, dim_head)  # 恢复形状

    # ✅ 计算最大误差（完整 attention 结果 softmax(QK^T) @ V）
    def max_diff(A, B, name_A, name_B):
        diff = (A - B).abs().max().item()
        print(f"Max error between {name_A} and {name_B}: {diff:.6f}")

    max_diff(attn_pytorch, attn_einsum, "PyTorch", "Einsum")
    max_diff(attn_pytorch, attn_matmul, "PyTorch", "Matmul")
    max_diff(attn_pytorch, attn_xformers, "PyTorch", "xFormers")
    max_diff(attn_einsum, attn_matmul, "Einsum", "Matmul")
    max_diff(attn_einsum, attn_xformers, "Einsum", "xFormers")
    max_diff(attn_matmul, attn_xformers, "Matmul", "xFormers")
