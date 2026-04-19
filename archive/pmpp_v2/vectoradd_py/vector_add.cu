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

    // TODO: wrap this launch with AT_DISPATCH_FLOATING_TYPES_AND_HALF
    // and use scalar_t instead of float

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "vector_add_kernel", ([&] {
        vector_add_kernel<scalar_t><<<blocks, threads>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}
