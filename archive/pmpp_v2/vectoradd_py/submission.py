#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu L4

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# Edit vector_add.cu locally for syntax highlighting.
# This string is what actually gets compiled on the remote GPU.
# To sync: copy your .cu contents here, or run the build script.
CUDA_SRC = r"""
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void vector_add_kernel(const scalar_t* A, const scalar_t* B, scalar_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = A[idx] + B[idx];
    }
}

torch::Tensor vector_add(torch::Tensor A, torch::Tensor B, torch::Tensor output) {
    int N = A.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "vector_add_kernel", ([&] {
        vector_add_kernel<scalar_t><<<blocks, threads>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}
"""

CPP_SRC = "torch::Tensor vector_add(torch::Tensor A, torch::Tensor B, torch::Tensor output);"

module = load_inline(
    name='vectoradd_v2',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['vector_add'],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    return module.vector_add(A, B, output)
