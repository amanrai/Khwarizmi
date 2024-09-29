import numpy as np
import time

import torch
size = 8192
a = np.random.uniform(0, 1, (size, size))
b = np.random.uniform(0,1, (size, size))

times = []
for i in range(100):
    start = time.time()
    r = np.matmul(a, b)
    end = time.time()
    times.append(end - start)
print("A: ", a.shape)
print("B: ", b.shape)
print("Numpy Time: ", np.mean(times)*1000, "ms")

x = torch.FloatTensor(a)
y = torch.FloatTensor(b)

times = []
for i in range(100):
    start = time.time()
    r = torch.matmul(x, y)
    end = time.time()
    times.append(end - start)
print("A: ", x.shape, x.device)
print("B: ", y.shape, y.device)
# print("Torch Time: ", (end - start)*1000, "ms")
print("Torch Time: ", np.mean(times)*1000, "ms")

_x = x.to("cuda")
_y = y.to("cuda")
times = []
for i in range(100):
    start = time.time()
    r = torch.matmul(_x, _y)
    end = time.time()
    times.append(end - start)
print("A: ", _x.shape, _x.device)
print("B: ", _y.shape, _y.device)
# print("Torch Time Cuda: ", (end - start)*1000, "ms")
print("Torch Time Cuda: ", np.mean(times)*1000, "ms")
