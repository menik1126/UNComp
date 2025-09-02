import torch
import triton
import triton.language as tl
import math

@triton.jit
def attention_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Matrix dimensions
    B, H, N, D,
    # Strides for different dimensions
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Scale for attention scores
    scale,
    BLOCK_SIZE: tl.constexpr):
    
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch index and head index
    batch_id = pid // (H * (N // BLOCK_SIZE))
    pid_tmp = pid % (H * (N // BLOCK_SIZE))
    head_id = pid_tmp // (N // BLOCK_SIZE)
    block_id = pid_tmp % (N // BLOCK_SIZE)
    
    # Base pointers for current batch and head
    q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
    k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + head_id * stride_vh
    o_base = o_ptr + batch_id * stride_ob + head_id * stride_oh
    
    # Start position for this block
    start_n = block_id * BLOCK_SIZE
    
    # Load query block
    offs_d = tl.arange(0, D)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N
    
    q_ptrs = q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask[:, None])
    
    # Initialize accumulator
    o = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)
    
    # Iterate over key/value sequence
    for k_start in range(0, N, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offs < N
        
        # Load key and value blocks
        k_ptrs = k_base + k_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + k_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=k_mask[:, None])
        v = tl.load(v_ptrs, mask=k_mask[:, None])
        
        # Compute attention scores and scale them
        scores = tl.dot(q, tl.trans(k))
        scores = scores * scale
        
        # Apply softmax
        scores = tl.softmax(scores, axis=1)
        
        # Compute weighted sum
        o += tl.dot(scores, v)
    
    # Write output
    o_ptrs = o_base + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=mask[:, None])

class TritonAttention(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.BLOCK_SIZE = 32
        
    def forward(self, query, key, value):
        batch_size, num_heads, seq_len, head_dim = query.shape
        assert head_dim == self.head_dim
        
        # Make sure inputs are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        
        # Allocate output tensor
        output = torch.empty_like(query)
        
        # Compute scale factor
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute number of blocks
        grid = (batch_size * num_heads * triton.cdiv(seq_len, self.BLOCK_SIZE),)
        
        # Launch kernel
        attention_kernel[grid](
            query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
            batch_size, num_heads, seq_len, head_dim,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            scale,
            self.BLOCK_SIZE
        )
        
        return output

def test_attention():
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 512
    head_dim = 64
    
    # Create test data
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    
    # Initialize attention module
    attention = TritonAttention(head_dim=head_dim)
    
    # PyTorch reference implementation
    def torch_attention(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, v)
    
    # Test both implementations
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Time PyTorch implementation
    start.record()
    torch_output = torch_attention(query, key, value)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end)
    
    # Time Triton implementation
    start.record()
    triton_output = attention(query, key, value)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end)
    
    # Print results
    print(f"Output shape: {triton_output.shape}")
    print(f"PyTorch time: {torch_time:.2f}ms")
    print(f"Triton time: {triton_time:.2f}ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    
    # Verify results
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)
    print("Output matches with PyTorch implementation!")

if __name__ == "__main__":
    test_attention()