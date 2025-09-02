import torch
import time

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream = torch.cuda.Stream()
event = torch.cuda.Event()
# 生成随机矩阵
x_cpu = torch.randn(30000, 30000)
x_gpu = torch.randn(30000, 30000, device=device)

# CPU 上的矩阵乘法
# start_cpu = time.time()
# y_cpu = x_cpu @ x_cpu
# end_cpu = time.time()

# GPU 上的矩阵乘法，执行 1000 次（可能是异步的）
start_gpu = time.time()
with torch.cuda.stream(stream): 
    for i in range(1):
        y_gpu = x_gpu @ x_gpu
    event.record(stream)
end_gpu = time.time()
# torch.cuda.current_stream().wait_stream(stream) 
# stream.synchronize()
# event.synchronize()
# torch.cuda.synchronize()
# 打印不使用同步的 GPU 时间
# print(f"CPU computation time: {end_cpu - start_cpu:.6f} seconds")
print(f"GPU computation time (without sync): {end_gpu - start_gpu:.6f} seconds")
# 打印 GPU 完全同步后的时间
print(f"Time after GPU synchronize: {time.time() - start_gpu:.6f} seconds")

# import torch
# import time

# # 初始化设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# stream = torch.cuda.Stream()

# # 创建两个事件用于计时
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# # 生成随机矩阵
# x_gpu = torch.randn(30000, 30000, device=device)

# # GPU 上的矩阵乘法，执行 1 次
# with torch.cuda.stream(stream): 
#     # 记录开始时间的事件
#     start_event.record(stream)
    
#     for i in range(1):
#         y_gpu = x_gpu @ x_gpu
    
#     # 记录结束时间的事件
#     end_event.record(stream)

# # 确保流中的所有操作完成
# stream.synchronize()

# # 计算 GPU 事件之间的时间
# elapsed_time = start_event.elapsed_time(end_event)

# # 打印 GPU 事件之间的时间
# print(f"GPU computation time (with event): {elapsed_time:.6f}   ")
