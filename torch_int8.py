import torch
import torch.utils.cpp_extension

cuda_source = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void int8_matmul_kernel(
    const char* __restrict__ A,
    const char* __restrict__ B,
    int* __restrict__ C,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += int(A[row * K + k]) * int(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}
'''

module = torch.utils.cpp_extension.load_inline(
    name='int8_matmul_cuda',
    cpp_sources=[],
    cuda_sources=[cuda_source],
    functions=['int8_matmul_kernel'],
    with_cuda=True,
    extra_cuda_cflags=['-O2'],
)

def int8_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
    assert A.dtype == torch.int8 and B.dtype == torch.int8, "Tensors must be int8"
    assert A.dim() == 2 and B.dim() == 2, "Tensors must be 2D"
    assert A.size(1) == B.size(0), "Inner dimensions must match"
    
    M, K = A.size()
    K, N = B.size()
    
    C = torch.empty(M, N, dtype=torch.int32, device='cuda')
    
    threads_per_block = 16
    blocks = (
        (N + threads_per_block - 1) // threads_per_block,
        (M + threads_per_block - 1) // threads_per_block
    )
    
    module.int8_matmul_kernel(
        grid=blocks,
        block=(threads_per_block, threads_per_block),
        args=[A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K]
    )
    
    return C

# Example usage
A = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8, device='cuda')
B = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8, device='cuda')

result = int8_matmul(A, B)
print(result.shape, result.dtype)