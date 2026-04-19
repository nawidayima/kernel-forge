#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu L4

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r"""
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void grayscale_kernel(const scalar_t* data, scalar_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 0.2989*data[idx*3+0] + 0.5870*data[idx*3+1] + 0.1140*data[idx*3+2];
    }
}

torch::Tensor grayscale(torch::Tensor data, torch::Tensor output) {
    int N = output.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "grayscale_kernel", ([&] {
        grayscale_kernel<scalar_t><<<blocks, threads>>>(data.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}
"""

CPP_SRC = "torch::Tensor grayscale(torch::Tensor data, torch::Tensor output);"

module = load_inline(
    name='grayscale_v2',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['grayscale'],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    return module.grayscale(data, output)
