import torch
import time

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成随机矩阵
x_cpu = torch.randn(30000, 30000)
x_gpu = torch.randn(1000, 1000, device=device)

# CPU 上的矩阵乘法
# start_cpu = time.time()
# y_cpu = x_cpu @ x_cpu
# end_cpu = time.time()

# GPU 上的矩阵乘法，执行 1000 次（可能是异步的）
start_gpu = time.time()
for i in range(10000):
    y_gpu = x_gpu @ x_gpu
end_gpu = time.time()
# torch.cuda.synchronize()
# 打印不使用同步的 GPU 时间
# print(f"CPU computation time: {end_cpu - start_cpu:.6f} seconds")
print(f"GPU computation time (without sync): {end_gpu - start_gpu:.6f} seconds")
# 打印 GPU 完全同步后的时间
print(f"Time after GPU synchronize: {time.time() - start_gpu:.6f} seconds")
