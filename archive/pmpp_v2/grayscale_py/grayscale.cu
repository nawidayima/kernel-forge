#include <cuda_runtime.h>

// YOUR KERNEL HERE
// Each thread converts one pixel: out[i] = 0.2989*R + 0.5870*G + 0.1140*B
// Think about: how is the RGB data laid out in memory?
// If pixel i has channels at data[i*3+0], data[i*3+1], data[i*3+2],
// what does that mean for memory access across adjacent threads?
template <typename scalar_t>
__global__ void grayscale_kernel(const scalar_t* data, scalar_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 0.2989*data[idx*3+0] + 0.5870*data[idx*3+1] + 0.1140*data[idx*3+2];
    }
}

// END YOUR KERNEL

torch::Tensor grayscale(torch::Tensor data, torch::Tensor output) {
    int N = output.numel();  // total number of pixels
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "grayscale_kernel", ([&] {
        // TODO: launch your kernel here
        grayscale_kernel<scalar_t><<<blocks, threads>>>(data.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}
