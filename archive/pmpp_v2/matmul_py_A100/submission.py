#!POPCORN leaderboard matmul_v2
#!POPCORN gpu A100

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r"""
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* A, const scalar_t* B, scalar_t* C, int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        float sum = 0;
        for (int k = 0; k < K; ++k){
            sum += float(A[row*K+k])*float(B[k*N+col]);
        }
        C[row * N + col] = static_cast<scalar_t>(sum);
    }
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matmul_kernel", ([&] {
        matmul_kernel<scalar_t><<<blocks, threads>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), M, K, N);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return c;
}
"""

CPP_SRC = "torch::Tensor matmul(torch::Tensor a, torch::Tensor b, torch::Tensor c);"

module = load_inline(
    name='matmul_v2',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['matmul'],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return module.matmul(a, b, c)
