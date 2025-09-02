import torch
from dkernel import SparseAttention, LocalStrideSparseAttention


# 1.) Using local-stride pattern

block_size = 32 # sparse block size, minimum 16
local_blocks = 32 # num local blocks, always attend to up to 64 * 16=1024 tokens
vert_stride = 8 # attend to 1 block per every 8 blocks after the local window above
max_seq_len = 8192 # model supports up to 8192 seqlen
num_heads = 32
device = "cuda"

q, k, v = [torch.rand(3, 8192, 32, 128,
                device=device).requires_grad_()
                for _ in range(3)]

# attn = LocalStrideSparseAttention(
#                  num_heads,
#                  max_seq_len,
#                  block_size,
#                  local_blocks,
#                  vert_stride,
#                  seq_dim=1, # q/k/v layout: (batch, seq, heads, head_dim)
#                 )
# attn.to(device) # optional, attn default to current_device

# # For the first time, it needs to warmup, so could be slow.
# attn(q, k, v)

# # Now should be fast
# ouput = attn(q, k, v)


# 2.) Using user defined arbitrary pattern

num_blocks = max_seq_len // block_size

# True/1 means attn to the blocks, 0 means not attend to.
block_sparse_pattern = torch.rand((num_heads, num_blocks, num_blocks)) > 0.8
# 形状: (num_heads, 128, 128)

# Ensure the diag blocks are always attended.
# Otherwise, tokens at block_0 have nothing to attend to, resulting in nan
for head_sparse_pattern in block_sparse_pattern:
    head_sparse_pattern.diagonal()[:] = True

# Ensure it is causal
block_sparse_pattern *= torch.tril(torch.ones_like(block_sparse_pattern[0]))

# NOTE: You may get warning saying that pattern is not KV cache efficient, due to
# KV cache needed for later tokens are not used in earlier tokens.
# This may result in unexpected larger KV cache.
# So you may need to re-consider the design of the sparse pattern.

attn = SparseAttention(block_size, block_sparse_pattern)
attn.to(device)

# similar, it needs to warmup for the first time
output = attn(q, k, v)

seq_lens = [10, 20, 15]
seqlens = torch.tensor(seq_lens)
cu_seqlen = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(seq_lens))])

# 推理时使用
output = attn(
    q, k, v,
    cu_seqlen_k=cu_seqlen,  # K的累积长度
    cu_seqlen_q=cu_seqlen,  # Q的累积长度
    seqlens=seqlens         # 实际序列长度
)