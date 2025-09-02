import torch
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {(end - start)*1000:.2f} ms")

def compute_without_streams(a, b, c, d, iterations=1000):
    """不使用CUDA流的计算"""
    for _ in range(iterations):
        # 模拟计算密集型操作
        x1 = torch.matmul(a, b)
        x2 = torch.matmul(c, d)
        # 强制同步
        torch.cuda.synchronize()
    return x1, x2

def compute_with_streams(a, b, c, d, iterations=1000):
    """使用CUDA流的并行计算"""
    # 创建两个独立的CUDA流
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    
    for _ in range(iterations):
        with torch.cuda.stream(s1):
            x1 = torch.matmul(a, b)
            
        with torch.cuda.stream(s2):
            x2 = torch.matmul(c, d)
            
    # 同步所有流
    torch.cuda.synchronize()
    return x1, x2

def main():
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 创建测试数据
    size = 1024
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    c = torch.randn(size, size, device='cuda')
    d = torch.randn(size, size, device='cuda')
    
    # 预热GPU
    print("Warming up GPU...")
    for _ in range(10):
        x = torch.matmul(a, b)
        y = torch.matmul(c, d)
    torch.cuda.synchronize()
    
    # 运行测试
    print("\nStarting benchmarks...")
    iterations = 100
    
    # 不使用流
    with timer("Without streams"):
        x1, x2 = compute_without_streams(a, b, c, d, iterations)
    
    # 清除GPU缓存
    torch.cuda.empty_cache()
    
    # 使用流
    with timer("With streams"):
        x1, x2 = compute_with_streams(a, b, c, d, iterations)
    
    # 验证结果一致性
    print("\nVerifying results...")
    # 重新计算一次用作验证
    with torch.no_grad():
        verify1 = torch.matmul(a, b)
        verify2 = torch.matmul(c, d)
    
    print(f"Results match for computation 1: {torch.allclose(x1, verify1)}")
    print(f"Results match for computation 2: {torch.allclose(x2, verify2)}")

if __name__ == "__main__":
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        main()